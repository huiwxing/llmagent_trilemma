"""
Energy modeling for large language models.

This module provides classes and functions for measuring and modeling energy consumption
of large language models based on input and output token counts.
"""

import numpy as np
from scipy import optimize
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Any, Dict
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


class UnifiedEnergyMonitor:
    """
    A unified energy monitoring class that measures energy consumption
    for CPU and GPU operations.
    """
    
    def __init__(self, cpu_sampling_interval: float = 0.5, include_idle: bool = False):
        """
        Initialize the energy monitor.
        
        Args:
            cpu_sampling_interval (float): Sampling interval for CPU measurements in seconds.
            include_idle (bool): Whether to include idle energy in the measurements.
        """
        self.cpu_sampling_interval = cpu_sampling_interval
        self.include_idle = include_idle
        
        # Import these here to avoid dependency issues when not using GPU
        try:
            import psutil
            import pynvml
            self.psutil = psutil
            self.pynvml = pynvml
            self.has_gpu = True
            try:
                self.pynvml.nvmlInit()
                self.device_count = self.pynvml.nvmlDeviceGetCount()
            except self.pynvml.NVMLError:
                self.has_gpu = False
                self.device_count = 0
        except ImportError:
            self.has_gpu = False
            self.device_count = 0
    
    def measure_energy(self, func: Callable[[], Any]) -> Tuple[Any, Dict[str, float]]:
        """
        Measure the energy consumption of a function.
        
        Args:
            func (Callable): The function to measure.
            
        Returns:
            Tuple containing the function result and energy measurements.
        """
        # Start monitoring
        cpu_energy = 0.0
        gpu_energy = 0.0
        
        # Setup CPU monitoring
        import threading
        import time
        
        stop_monitoring = threading.Event()
        
        def cpu_monitor():
            nonlocal cpu_energy
            start_time = time.time()
            last_measurement_time = start_time
            
            if hasattr(self, 'psutil'):
                # Get the initial CPU power consumption
                initial_cpu_power = self.psutil.cpu_percent(interval=None) / 100.0 * 100.0  # Rough estimate: 100W for full CPU
                
                while not stop_monitoring.is_set():
                    time.sleep(self.cpu_sampling_interval)
                    current_time = time.time()
                    interval = current_time - last_measurement_time
                    
                    # Get CPU power consumption
                    cpu_power = self.psutil.cpu_percent(interval=None) / 100.0 * 100.0
                    
                    # Calculate energy (power * time)
                    if self.include_idle or cpu_power > initial_cpu_power:
                        cpu_energy += cpu_power * interval
                    
                    last_measurement_time = current_time
        
        def gpu_monitor():
            nonlocal gpu_energy
            start_time = time.time()
            last_measurement_time = start_time
            
            if self.has_gpu and self.device_count > 0:
                # Get initial GPU power for all devices
                initial_gpu_powers = []
                handles = []
                
                for i in range(self.device_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    handles.append(handle)
                    
                    try:
                        power = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                        initial_gpu_powers.append(power)
                    except self.pynvml.NVMLError:
                        initial_gpu_powers.append(0.0)
                
                while not stop_monitoring.is_set():
                    time.sleep(self.cpu_sampling_interval)  # Use same interval as CPU
                    current_time = time.time()
                    interval = current_time - last_measurement_time
                    
                    # Get GPU power for all devices
                    for i, handle in enumerate(handles):
                        try:
                            power = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                            
                            # Calculate energy (power * time)
                            if self.include_idle or power > initial_gpu_powers[i]:
                                gpu_energy += power * interval
                        except self.pynvml.NVMLError:
                            pass
                    
                    last_measurement_time = current_time
        
        # Start monitoring threads
        cpu_thread = threading.Thread(target=cpu_monitor)
        gpu_thread = threading.Thread(target=gpu_monitor)
        
        cpu_thread.daemon = True
        gpu_thread.daemon = True
        
        cpu_thread.start()
        gpu_thread.start()
        
        # Run the function
        start_time = time.time()
        result = func()
        end_time = time.time()
        
        # Stop monitoring
        stop_monitoring.set()
        cpu_thread.join(timeout=1.0)
        gpu_thread.join(timeout=1.0)
        
        # Return the results
        measurements = {
            'cpu_energy': cpu_energy,
            'gpu_energy': gpu_energy,
            'total_energy': cpu_energy + gpu_energy,
            'execution_time': end_time - start_time
        }
        
        return result, measurements


@dataclass
class EnergyMeasurement:
    """
    Data class for storing energy measurement results.
    
    Attributes:
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
        energy_consumption (float): Measured energy consumption.
    """
    input_tokens: int
    output_tokens: int
    energy_consumption: float


class LLMEnergyModel:
    """
    Model for fitting and predicting LLM energy consumption.
    
    This class implements a model that relates the number of input and output tokens
    to energy consumption using the formula:
    
    e_K(τin, τout) = αK,0*τin + αK,1*τout + αK,2*τin*τout
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
                 alpha_0: float = None, 
                 alpha_1: float = None, 
                 alpha_2: float = None,
                 tokenizer = None,
                 model = None):
        """
        Initialize the LLM energy model.
        
        Args:
            model_name (str): Name of the language model.
            alpha_0 (float, optional): Coefficient for input tokens.
            alpha_1 (float, optional): Coefficient for output tokens.
            alpha_2 (float, optional): Coefficient for the interaction term.
            tokenizer (optional): Pre-loaded tokenizer. If None, will be loaded from model_name.
            model (optional): Pre-loaded model. If None, needs to be set before generation.
        """
        self.model_name = model_name
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        
        # If tokenizer is provided, use it. Otherwise, load from model_name
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name)
        
        # Store the model
        self.model = model

    def _energy_function(self, 
                         params: np.ndarray,
                         input_tokens: np.ndarray,
                         output_tokens: np.ndarray) -> np.ndarray:
        """
        Energy consumption function: e_K(τin, τout) = αK,0*τin + αK,1*τout + αK,2*τin*τout
        
        Args:
            params (np.ndarray): Parameters [alpha_0, alpha_1, alpha_2].
            input_tokens (np.ndarray): Number of input tokens.
            output_tokens (np.ndarray): Number of output tokens.
            
        Returns:
            np.ndarray: Predicted energy consumption.
        """
        alpha_0, alpha_1, alpha_2 = params
        return (alpha_0 * input_tokens +
                alpha_1 * output_tokens +
                alpha_2 * input_tokens * output_tokens)

    def _error_function(self, 
                        params: np.ndarray,
                        input_tokens: np.ndarray,
                        output_tokens: np.ndarray,
                        measured_energy: np.ndarray) -> float:
        """
        Calculate mean squared error between predicted and measured energy.
        
        Args:
            params (np.ndarray): Parameters [alpha_0, alpha_1, alpha_2].
            input_tokens (np.ndarray): Number of input tokens.
            output_tokens (np.ndarray): Number of output tokens.
            measured_energy (np.ndarray): Measured energy consumption.
            
        Returns:
            float: Mean squared error.
        """
        predicted = self._energy_function(params, input_tokens, output_tokens)
        return np.mean((predicted - measured_energy) ** 2)

    def generate_text(self,
                      prompt: str,
                      target_length: int,
                      max_length: int = 2048) -> str:
        """
        Generate text with approximately target number of tokens.
        
        Args:
            prompt (str): Input prompt.
            target_length (int): Target number of output tokens.
            max_length (int): Maximum length for generation.
            
        Returns:
            str: Generated text.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set model before generation.")
            
        output_ids = self.model(
            prompt,
            max_tokens=target_length + 1,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1
        )

        return output_ids[0]['generated_text']

    async def measure_energy(self,
                             input_text: str,
                             target_output_length: int) -> Tuple[int, int, float]:
        """
        Measure energy consumption for a single LLM call.
        
        Args:
            input_text (str): Input text.
            target_output_length (int): Target length of the output.
            
        Returns:
            Tuple: Input tokens, output tokens, and energy consumption.
        """
        monitor = UnifiedEnergyMonitor(cpu_sampling_interval=0.5, include_idle=True)

        input_tokens = len(self.tokenizer.encode(input_text))

        output_text, measurements = monitor.measure_energy(
            lambda: self.generate_text(input_text, target_output_length))

        output_tokens = len(self.tokenizer.encode(output_text))
        return input_tokens, output_tokens, measurements['total_energy']

    async def collect_measurements(self, data_file_path: Optional[str] = None) -> List[EnergyMeasurement]:
        """
        Collect energy measurements with varying input and output lengths.
        
        Args:
            data_file_path (str, optional): Path to a JSON data file to use for prompts.
            
        Returns:
            List[EnergyMeasurement]: List of energy measurements.
        """
        measurements = []

        input_lengths = [500 * i for i in range(1, 10)]
        output_lengths = [50 * i for i in range(1, 10)]

        # Load data for prompts if provided
        if data_file_path:
            import json
            with open(data_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            json_str = json.dumps(raw_data[:200])
            current_tokens = self.tokenizer.encode(json_str)
            current_length = len(current_tokens)
            print(f"Loaded {current_length} tokens from data file")
        else:
            # Generate synthetic prompt
            lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100
            current_tokens = self.tokenizer.encode(lorem_ipsum)
            current_length = len(current_tokens)
            print(f"Generated synthetic prompt with {current_length} tokens")

        for output_length in output_lengths:
            for input_length in input_lengths:
                if current_length >= input_length:
                    prompt_tokens = current_tokens[:input_length - 1]
                    prompt = self.tokenizer.decode(prompt_tokens)

                    # Measure energy consumption
                    input_tokens, output_tokens, energy = await self.measure_energy(
                        prompt, output_length
                    )

                    measurements.append(EnergyMeasurement(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        energy_consumption=energy
                    ))
                    print(measurements[-1])

        return measurements

    def fit(self, measurements: List[EnergyMeasurement]) -> bool:
        """
        Fit the energy consumption model using collected measurements.
        
        Args:
            measurements (List[EnergyMeasurement]): List of energy measurements.
            
        Returns:
            bool: True if fitting succeeded, False otherwise.
        """
        input_tokens = np.array([m.input_tokens for m in measurements])
        output_tokens = np.array([m.output_tokens for m in measurements])
        energy_values = np.array([m.energy_consumption for m in measurements])

        # IQR method to detect outliers
        def detect_outliers_iqr(X):
            Q1 = np.percentile(X, 25)
            Q3 = np.percentile(X, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (X >= lower_bound) & (X <= upper_bound)

        # Apply outlier detection to energy values
        energy_mask = detect_outliers_iqr(energy_values)

        # Print outlier information
        outliers = [(in_tok, out_tok, energy)
                    for in_tok, out_tok, energy, is_valid in
                    zip(input_tokens, output_tokens, energy_values, energy_mask)
                    if not is_valid]

        print(f"\nDetected {len(outliers)} outliers:")
        for in_tok, out_tok, energy in outliers:
            print(f"Input: {in_tok}, Output: {out_tok}, Energy: {energy:.2f}")

        # Remove outliers
        input_tokens = input_tokens[energy_mask]
        output_tokens = output_tokens[energy_mask]
        energy_values = energy_values[energy_mask]

        print(f"\nFitting with {len(input_tokens)} measurements after outlier removal")

        # Add non-negative constraints
        bounds = [(0, None), (0, None), (0, None)]  # All coefficients must be non-negative

        # Optimize
        result = optimize.minimize(
            self._error_function,
            [0.001, 0.001, 0.001],
            args=(input_tokens, output_tokens, energy_values),
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            self.alpha_0, self.alpha_1, self.alpha_2 = result.x

            # Calculate fitting statistics
            predicted_values = self._energy_function(result.x, input_tokens, output_tokens)

            # R-squared (coefficient of determination)
            ss_total = np.sum((energy_values - np.mean(energy_values)) ** 2)
            ss_residual = np.sum((energy_values - predicted_values) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            # Adjusted R-squared
            n = len(energy_values)
            p = 3  # Number of parameters
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

            # Mean squared error (MSE) and root mean squared error (RMSE)
            mse = np.mean((energy_values - predicted_values) ** 2)
            rmse = np.sqrt(mse)

            # Mean absolute error (MAE)
            mae = np.mean(np.abs(energy_values - predicted_values))

            # Calculate 95% confidence interval
            from scipy import stats
            confidence_level = 0.95
            degrees_of_freedom = n - p
            t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

            # Standard error
            residuals = energy_values - predicted_values
            std_error = np.sqrt(np.sum(residuals ** 2) / degrees_of_freedom)

            # Print fitting statistics
            print("\nFitting Statistics:")
            print(f"R-squared: {r_squared:.4f}")
            print(f"Adjusted R-squared: {adjusted_r_squared:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Standard Error: {std_error:.4f}")
            print(f"95% Confidence Interval: ±{t_value * std_error:.4f}")

            return True
        
        return False

    def predict_energy(self, input_tokens: int, output_tokens: int) -> float:
        """
        Predict energy consumption using fitted coefficients.
        
        Args:
            input_tokens (int): Number of input tokens.
            output_tokens (int): Number of output tokens.
            
        Returns:
            float: Predicted energy consumption.
        """
        if any(coef is None for coef in [self.alpha_0, self.alpha_1, self.alpha_2]):
            raise ValueError("Model coefficients not fitted yet")

        return self._energy_function(
            [self.alpha_0, self.alpha_1, self.alpha_2],
            np.array([input_tokens]),
            np.array([output_tokens])
        )[0]

    def get_coefficients(self) -> Tuple[float, float, float]:
        """
        Return fitted coefficients.
        
        Returns:
            Tuple[float, float, float]: The three coefficients [alpha_0, alpha_1, alpha_2].
        """
        if any(coef is None for coef in [self.alpha_0, self.alpha_1, self.alpha_2]):
            raise ValueError("Model coefficients not fitted yet")
            
        return self.alpha_0, self.alpha_1, self.alpha_2