#!/usr/bin/env python3
"""
Improved Testbench with JSON mode support and better error handling
"""

import os
import json
import csv
import jsonlines
import yaml
import requests
import time
from tqdm import tqdm
from logger import Logger
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from typing import Tuple, Dict, Any, Optional


class ImprovedModelScorer:
    """Base scorer with improved error handling"""
    
    model_id = None
    model_type = None

    def check_everything(self, response_from_ai, benchmark_data):
        """Enhanced validation with better error reporting"""
        if response_from_ai is None:
            Logger.error(f"[X] Empty or None response from model")
            return False, False, False, 0

        # Handle both string and dict responses
        if isinstance(response_from_ai, dict):
            json_response_from_ai = response_from_ai
        elif isinstance(response_from_ai, str):
            if response_from_ai.strip() == "":
                Logger.error(f"[X] Empty response from model")
                return False, False, False, 0
            try:
                json_response_from_ai = json.loads(response_from_ai)
            except (json.JSONDecodeError, TypeError) as error:
                Logger.error(f"[X] JSON parsing error: {error}")
                Logger.error(f"[X] Raw response (first 300 chars): {str(response_from_ai)[:300]}...")
                return False, False, False, 0
        else:
            Logger.error(f"[X] Unexpected response type: {type(response_from_ai)}")
            return False, False, False, 0

        is_json_good = self.validate_json(json_response_from_ai, self.get_json_schema())
        if not is_json_good:
            return False, False, False, 0

        correct_plugin = benchmark_data["messages"][-2]["expected_plugins"]
        is_plugin_correct = self.validate_plugin(json_response_from_ai, correct_plugin)
        if not is_plugin_correct:
            return True, False, False, 0

        correct_command = benchmark_data["messages"][-1]["binaries_used"]
        is_command_running = self.validate_command(json_response_from_ai, correct_command)
        
        return True, is_plugin_correct, is_command_running, 1

    @staticmethod
    def validate_json(content, schema):
        try:
            validate(instance=content, schema=schema)
            Logger.info(f"[✓] JSON validation passed")
            return True
        except ValidationError as e:
            Logger.error(f"[X] Schema validation failed: {e.message}")
            Logger.error(f"    Path: {' -> '.join(str(p) for p in e.path)}")
            return False
        except json.JSONDecodeError as e:
            Logger.error(f"[X] JSON parsing error: {e.msg}")
            return False

    @staticmethod
    def validate_plugin(gpt_response, correct_plugin):
        for plugin in gpt_response.get("commands", []):
            plugin_name = plugin.get("plugin_name", "")
            Logger.info(f"[*] Expected plugin: {correct_plugin}, Got: {plugin_name}")
            if plugin_name == correct_plugin:
                Logger.info(f"[✓] Plugin match!")
                return True
        Logger.error(f"[X] Plugin mismatch - expected {correct_plugin}")
        return False

    @staticmethod
    def validate_command(gpt_response, correct_command):
        command_map = []
        for plugin in gpt_response.get("commands", []):
            if plugin.get("plugin_name") == "run_bash":
                cmd = plugin.get("args", {}).get("command", "")
                Logger.info(f"[*] Expected binaries: {correct_command}")
                Logger.info(f"[*] Got command: {cmd}")
                result = any(binary in cmd for binary in correct_command)
                command_map.append(result)
                if result:
                    Logger.info(f"[✓] Command contains expected binary!")
                else:
                    Logger.error(f"[X] Command missing expected binary")
            else:
                # Non-bash plugins automatically pass command check
                command_map.append(True)

        return all(command_map) if command_map else False

    @staticmethod
    def get_json_schema():
        return {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "criticism": {"type": "string"},
                        "speak": {"type": "string"},
                    },
                    "required": ["text", "reasoning", "criticism", "speak"],
                },
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "plugin_name": {
                                "type": "string",
                                "enum": [
                                    "run_bash",
                                    "generic_response",
                                    "netcat_listener",
                                    "msfvenom_payload",
                                    "google",
                                ],
                            },
                            "args": {"type": "object"},
                            "file_name": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["plugin_name", "args"],
                    },
                },
            },
            "required": ["thoughts", "commands"],
        }

    def process_with_model(self, benchmark_data: dict):
        raise NotImplementedError


class ImprovedCustomScorer(ImprovedModelScorer):
    """Enhanced CustomModelScorer with json_mode support"""
    
    def __init__(
        self,
        endpoint_url,
        model_id="custom",
        model_type="custom",
        temperature=0.7,
        max_tokens=1024,
        custom_payload=None,
        enable_json_mode=True,
    ):
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.custom_payload = custom_payload or {}
        self.enable_json_mode = enable_json_mode

    def process_with_model(self, benchmark_data):
        system_prompt = benchmark_data["messages"][0]["content"]
        user_prompt = benchmark_data["messages"][1]["content"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "top_p": 0.95,
            "top_k": 64,
        }

        # Add json_mode if enabled
        if self.enable_json_mode:
            payload["json_mode"] = True

        # Merge custom payload
        if self.custom_payload:
            payload.update(self.custom_payload)

        start_time = time.time()
        Logger.info(f"[*] Running model: {self.model_id}")
        Logger.info(f"[*] JSON mode: {self.enable_json_mode}")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            }
            
            response = requests.post(
                self.endpoint_url, 
                json=payload, 
                headers=headers,
                timeout=120  # Increased timeout for larger models
            )
            response.raise_for_status()
            response_data = response.json()

            Logger.info(f"[*] RAW RESPONSE: {json.dumps(response_data, indent=2)[:500]}...")

            # Handle different response formats
            model_response = self._extract_response(response_data)

        except requests.Timeout:
            Logger.error(f"[X] Request timeout after 120s")
            return None, None
        except requests.RequestException as e:
            Logger.error(f"[X] Request failed: {e}")
            return None, None
        except (KeyError, IndexError) as e:
            Logger.error(f"[X] Unexpected response format: {e}")
            return None, None

        end_time = time.time()
        response_time = end_time - start_time
        
        Logger.info(f"[✓] Response received in {response_time:.2f}s")
        return model_response, response_time

    def _extract_response(self, response_data: dict) -> Any:
        """Extract model response from various API formats"""
        
        # OpenAI-compatible format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            model_response = response_data["choices"][0]["message"]["content"]
            try:
                # Try to parse as JSON
                return json.loads(model_response.strip())
            except json.JSONDecodeError:
                return self._extract_json_from_text(model_response)
        
        # Custom format with "output"
        if "output" in response_data:
            output = response_data["output"]
            if isinstance(output, dict):
                return output
            if isinstance(output, str):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return self._extract_json_from_text(output)
        
        # Fallback to "response" field
        if "response" in response_data:
            resp = response_data["response"]
            if isinstance(resp, dict):
                return resp
            if isinstance(resp, str):
                try:
                    return json.loads(resp)
                except json.JSONDecodeError:
                    return self._extract_json_from_text(resp)
        
        raise ValueError("Could not extract response from API response")

    def _extract_json_from_text(self, text: str) -> Any:
        """Extract JSON from text with various markers"""
        import re
        
        # Remove thinking markers if present
        marker = "\n</think>\n\n"
        if marker in text:
            text = text.split(marker, 1)[1]
        
        # Try to extract JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Try markdown code blocks
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Return as string if all else fails
        Logger.warning(f"[!] Could not extract valid JSON, returning raw text")
        return text


class ScoreProcessor:
    """Score processing with enhanced metrics"""
    
    def __init__(self):
        self.scores = []

    def calculate_metrics(self, scores):
        Logger.info(f"[*] Calculating metrics for {len(scores)} scores")

        total_tests = len(scores)
        if total_tests == 0:
            return 0, 0, 0, 0

        valid_scores = [s for s in scores if len(s) == 5 and s[4] is not None]
        if len(valid_scores) == 0:
            return 0, 0, 0, 0

        # Structural Accuracy (valid JSON)
        structural_accuracy = sum(int(s[1]) for s in valid_scores) / total_tests

        # Functional Correctness (plugin + command correct)
        functional_correctness = sum(int(s[2] and s[3]) for s in valid_scores) / total_tests

        # Consistency (all checks pass)
        consistency_scores = [int(s[1] and s[2] and s[3]) for s in valid_scores]
        consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0

        # Average Response Time
        avg_response_time = sum(s[4] for s in valid_scores) / len(valid_scores)

        return structural_accuracy, functional_correctness, consistency, avg_response_time

    def process_scores(self, model_type, model_id, score):
        if len(score) > 5:
            score = score[:5]
        elif len(score) < 5:
            while len(score) < 5:
                score.append(None)

        model_found = False
        for sc in self.scores:
            if model_type in sc.keys():
                if model_id not in sc[model_type]:
                    sc[model_type][model_id] = []
                sc[model_type][model_id].append(score)
                model_found = True
                break

        if not model_found:
            self.scores.append({model_type: {model_id: [score]}})

    def write_model_scores_to_csv(self, model_type, model_id, output_dir="scores"):
        os.makedirs(output_dir, exist_ok=True)
        detailed_file = os.path.join(output_dir, f"{model_type}_{model_id}_detailed.csv")
        summary_file = os.path.join(output_dir, f"{model_type}_{model_id}_summary.csv")

        # Write detailed scores
        with open(detailed_file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                "Model Type", "Model ID", "Line Number",
                "isJsonValid", "isPluginValid", "isCommandCorrect", "ResponseTime"
            ])
            for sc in self.scores:
                if model_type in sc and model_id in sc[model_type]:
                    for score in sc[model_type][model_id]:
                        csvwriter.writerow([model_type, model_id] + score)

        # Write summary
        with open(summary_file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                "Model Type", "Model ID", "Structural Accuracy",
                "Functional Correctness", "Consistency", "Avg Response Time"
            ])
            for sc in self.scores:
                if model_type in sc and model_id in sc[model_type]:
                    scores = sc[model_type][model_id]
                    metrics = self.calculate_metrics(scores)
                    csvwriter.writerow([model_type, model_id] + list(metrics))

        Logger.info(f"[✓] Scores written to {output_dir}")

    def print_scores(self, line_number, json_check, plugin_check, command_check, model_type, response_time):
        status = "✓" if (json_check and plugin_check and command_check) else "X"
        Logger.info(
            f"[{status}] Line {line_number} | Model: {model_type} | "
            f"JSON: {json_check} | Plugin: {plugin_check} | Command: {command_check} | "
            f"Time: {response_time:.2f}s"
        )

    def run_model_benchmarks(self, scorer: ImprovedModelScorer, config, evaluation_file):
        line_number = 0
        with jsonlines.open(evaluation_file) as reader:
            for benchmark_data in tqdm(reader, desc=f"Testing {scorer.model_id}"):
                line_number += 1
                gpt_response, response_time = scorer.process_with_model(benchmark_data)
                json_check, plugin_check, command_check, _ = scorer.check_everything(
                    gpt_response, benchmark_data
                )
                self.print_scores(
                    line_number, json_check, plugin_check, command_check,
                    scorer.model_id, response_time if response_time else 0
                )
                self.process_scores(
                    scorer.model_type, scorer.model_id,
                    [line_number, json_check, plugin_check, command_check, response_time if response_time else 0]
                )


def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def main():
    # Load configuration
    config = load_config("improved_config.yaml")
    
    # Setup logging
    Logger.setup_logging(config.get("log_file", "improved_testbench.log"))
    
    # Evaluation file
    evaluation_file = config.get("testing_file", "evaluation/combined.jsonl")
    
    if not os.path.exists(evaluation_file):
        Logger.error(f"[X] Evaluation file not found: {evaluation_file}")
        return
    
    # Initialize score processor
    score_processor = ScoreProcessor()
    
    # Define scorers for all your models
    scorers = []
    
    # Add your model configurations here
    model_configs = config.get("models", [])

    print(model_configs)
    
    for model_config in model_configs:
        if not model_config.get("enabled", True):
            continue
            
        scorer = ImprovedCustomScorer(
            endpoint_url=model_config["endpoint"],
            model_id=model_config["model_id"],
            model_type=model_config["model_type"],
            temperature=model_config.get("temperature", 0.7),
            max_tokens=model_config.get("max_tokens", 1024),
            custom_payload=model_config.get("custom_payload", {}),
            enable_json_mode=model_config.get("json_mode", True),
        )
        scorers.append(scorer)
    
    # Run benchmarks
    for scorer in scorers:
        Logger.info(f"\n{'='*60}")
        Logger.info(f"Testing model: {scorer.model_id}")
        Logger.info(f"{'='*60}\n")
        
        score_processor.run_model_benchmarks(scorer, config, evaluation_file)
        score_processor.write_model_scores_to_csv(
            scorer.model_type, scorer.model_id, config.get("output_dir", "scores")
        )
        
        time.sleep(2)  # Brief pause between models
    
    Logger.info(f"\n{'='*60}")
    Logger.info("✅ Benchmarking complete!")
    Logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
