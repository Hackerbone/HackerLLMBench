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
from openai import OpenAI


class ModelScorer:
    model_id = None
    model_type = None

    def check_everything(self, response_from_ai, benchmark_data):
        try:
            json_response_from_ai = json.loads(response_from_ai)
        except (json.JSONDecodeError, TypeError) as error:
            Logger.error(f"[X] Some JSON error has occurred: {error}")
            return False, False, False, 0  # Added 0 for response time placeholder

        is_json_good = self.validate_json(json_response_from_ai, self.get_json_schema())
        if not is_json_good:
            return False, False, False, 0  # Added 0 for response time placeholder

        correct_plugin = benchmark_data["messages"][-2]["expected_plugins"]
        is_plugin_correct = self.validate_plugin(json_response_from_ai, correct_plugin)
        if not is_plugin_correct:
            return True, False, False, 0  # Added 0 for response time placeholder

        correct_command = benchmark_data["messages"][-1]["binaries_used"]
        is_command_running = self.validate_command(
            json_response_from_ai, correct_command
        )
        return (
            True,
            is_plugin_correct,
            is_command_running,
            1,
        )  # Added 1 for response time placeholder

    @staticmethod
    def validate_json(content, schema):
        try:
            validate(instance=content, schema=schema)
            return True
        except ValidationError as e:
            Logger.error(f"Validation failed: {e.message}")
            return False
        except json.JSONDecodeError as e:
            Logger.error(f"JSON parsing error: {e.msg}")
            return False

    @staticmethod
    def validate_plugin(gpt_response, correct_plugin):
        for plugin in gpt_response["commands"]:
            Logger.info(
                f"[*] Correct plugin: {correct_plugin}, Got Plugin from GPT: {plugin['plugin_name']}"
            )
            if plugin["plugin_name"] == correct_plugin:
                return True
        return False

    @staticmethod
    def validate_command(gpt_response, correct_command):
        command_map = []
        for plugin in gpt_response["commands"]:
            if plugin["plugin_name"] == "run_bash":
                Logger.info(
                    f"[*] Correct command: {correct_command}, Got command from GPT: {plugin['args']['command']}"
                )
                result = any(
                    command in plugin["args"]["command"] for command in correct_command
                )
                command_map.append(result)
            else:
                command_map.append(True)

        return all(command_map)

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
                        "allOf": [
                            {
                                "if": {
                                    "properties": {"plugin_name": {"const": "run_bash"}}
                                },
                                "then": {
                                    "properties": {"args": {"required": ["command"]}},
                                    "required": ["args"],
                                },
                            },
                            {
                                "if": {
                                    "properties": {
                                        "plugin_name": {"const": "generic_response"}
                                    }
                                },
                                "then": {
                                    "properties": {"args": {"required": ["response"]}},
                                    "required": ["args"],
                                },
                            },
                            {
                                "if": {
                                    "properties": {
                                        "plugin_name": {"const": "netcat_listener"}
                                    }
                                },
                                "then": {
                                    "properties": {"args": {"required": ["lport"]}},
                                    "required": ["args"],
                                },
                            },
                            {
                                "if": {
                                    "properties": {
                                        "plugin_name": {"const": "msfvenom_payload"}
                                    }
                                },
                                "then": {
                                    "properties": {
                                        "args": {
                                            "required": [
                                                "lhost",
                                                "lport",
                                                "payload",
                                                "file_format",
                                                "file_name",
                                            ]
                                        },
                                        "file_name": {"minItems": 1},
                                    },
                                    "required": ["args", "file_name"],
                                },
                            },
                        ],
                    },
                },
            },
            "required": ["thoughts", "commands"],
        }

    def process_with_model(self, benchmark_data: dict):
        raise NotImplementedError


class ScoreProcessor:
    def __init__(self):
        self.scores = []

    def calculate_metrics(self, scores):
        Logger.info(f"[*] Calculating metrics for {len(scores)} scores")
        Logger.info(f"[*] Scores: {scores}")

        total_tests = len(scores)
        if total_tests == 0:
            return 0, 0, 0, 0  # Modified to include 4th metric

        # Ensure that all scores have exactly five elements
        valid_scores = [
            score for score in scores if len(score) == 5 and score[4] is not None
        ]

        if len(valid_scores) == 0:
            return 0, 0, 0, 0  # Modified to include 4th metric

        # Calculate Structural Accuracy
        structural_accuracy = (
            sum([int(score[1]) for score in valid_scores]) / total_tests
        )

        # Calculate Functional Correctness
        functional_correctness = (
            sum([int(score[2] and score[3]) for score in valid_scores]) / total_tests
        )

        # Calculate Consistency: Average score across all runs
        consistency_scores = [
            int(score[1] and score[2] and score[3]) for score in valid_scores
        ]
        consistency = sum(consistency_scores) / len(consistency_scores)

        # Calculate Average Response Time
        avg_response_time = sum([score[4] for score in valid_scores]) / len(
            valid_scores
        )

        return (
            structural_accuracy,
            functional_correctness,
            consistency,
            avg_response_time,
        )

    def process_scores(self, model_type, model_id, score):
        model_found = False

        # Ensure the score list has exactly five elements
        if len(score) > 5:
            score = score[:5]
        elif len(score) < 5:
            # If fewer than 5, append None or another appropriate default value
            while len(score) < 5:
                score.append(None)

        # Loop through the existing scores to find the model_type
        for sc in self.scores:
            if model_type in sc.keys():
                if model_id not in sc[model_type]:
                    sc[model_type][model_id] = []  # Initialize the list if not present
                sc[model_type][model_id].append(score)
                model_found = True
                break

        if not model_found:
            # If the model_type is not found, add it to the scores list
            self.scores.append({model_type: {model_id: [score]}})

    def write_model_scores_to_csv(self, model_type, model_id, output_dir="scores"):
        os.makedirs(output_dir, exist_ok=True)
        detailed_file = os.path.join(
            output_dir, f"{model_type}_{model_id}_detailed.csv"
        )
        summary_file = os.path.join(output_dir, f"{model_type}_{model_id}_summary.csv")

        # Write detailed file as usual
        with open(detailed_file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "Model Type",
                    "Model ID",
                    "Line Number",
                    "isJsonValid",
                    "isPluginValid",
                    "isCommandCorrect",
                    "ResponseTime",
                ]
            )
            for sc in self.scores:
                if model_type in sc and model_id in sc[model_type]:
                    for score in sc[model_type][model_id]:
                        csvwriter.writerow([model_type, model_id] + score)

        # Read existing summary data if it exists
        existing_scores = {}
        if os.path.exists(summary_file):
            with open(summary_file, "r") as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # Skip header
                for row in csvreader:
                    existing_model_type = row[0]
                    existing_model_id = row[1]
                    existing_scores[(existing_model_type, existing_model_id)] = {
                        "Structural Accuracy": float(row[2]),
                        "Functional Correctness": float(row[3]),
                        "Consistency": float(row[4]),
                        "Avg Response Time": float(row[5]),
                    }

        # Calculate new metrics
        with open(summary_file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "Model Type",
                    "Model ID",
                    "Structural Accuracy",
                    "Functional Correctness",
                    "Consistency",
                    "Avg Response Time",
                ]
            )
            for sc in self.scores:
                if model_type in sc and model_id in sc[model_type]:
                    scores = sc[model_type][model_id]
                    (
                        structural_accuracy,
                        functional_correctness,
                        new_consistency,
                        avg_response_time,
                    ) = self.calculate_metrics(scores)

                    # Update consistency with averaging if exists
                    key = (model_type, model_id)
                    if key in existing_scores:
                        print("Key found, updating consistency")
                        existing_consistency = existing_scores[key]["Consistency"]
                        updated_consistency = (
                            existing_consistency + new_consistency
                        ) / 2
                    else:
                        print("Key not found, using new consistency")
                        updated_consistency = new_consistency

                    avg_response_time_normalized = (
                        avg_response_time  # Normalizing response time
                    )
                    csvwriter.writerow(
                        [
                            model_type,
                            model_id,
                            structural_accuracy,
                            functional_correctness,
                            updated_consistency,
                            avg_response_time_normalized,
                        ]
                    )

    def write_aggregate_summary_to_csv(self, output_dir="scores"):
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "aggregate_summary.csv")

        with open(summary_file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "Model Type",
                    "Model ID",
                    "Structural Accuracy",
                    "Functional Correctness",
                    "Consistency",
                    "Avg Response Time",
                ]
            )
            for model_score in self.scores:
                model_type = list(model_score.keys())[0]
                model_id = list(model_score[model_type].keys())[0]
                scores = model_score[model_type][model_id]

                (
                    structural_accuracy,
                    functional_correctness,
                    consistency,
                    avg_response_time,
                ) = self.calculate_metrics(scores)
                avg_response_time_normalized = (
                    avg_response_time  # Normalizing response time
                )

                csvwriter.writerow(
                    [
                        model_type,
                        model_id,
                        structural_accuracy,
                        functional_correctness,
                        consistency,
                        avg_response_time_normalized,
                    ]
                )

    def print_scores(
        self,
        line_number,
        json_check,
        plugin_check,
        command_check,
        model_type,
        response_time,
    ):
        Logger.info(
            f"---> For model {model_type} in JSONL line number {line_number}, isJsonValid: {json_check}, "
            f"isPluginValid: {plugin_check}, isCommandCorrect: {command_check}, ResponseTime: {response_time}"
        )
        Logger.info("-----------------------X--------------------------\n")

    def run_model_benchmarks(self, scorer: ModelScorer, config, evaluation_file):
        line_number = 0

        with jsonlines.open(evaluation_file) as reader:
            for benchmark_data in tqdm(reader, desc="Processing benchmarks"):
                line_number += 1
                gpt_response, response_time = scorer.process_with_model(benchmark_data)
                json_check, plugin_check, command_check, _ = scorer.check_everything(
                    gpt_response, benchmark_data
                )
                self.print_scores(
                    line_number,
                    json_check,
                    plugin_check,
                    command_check,
                    scorer.model_id,
                    response_time,
                )
                self.process_scores(
                    scorer.model_type,
                    scorer.model_id,
                    [
                        line_number,
                        json_check,
                        plugin_check,
                        command_check,
                        response_time,
                    ],
                )


class OpenAIScorer(ModelScorer):
    def __init__(self, api_key=None, model_id="gpt-4o", model_type="openai"):
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.model_type = model_type

    def process_with_model(self, benchmark_data):
        system_prompt = benchmark_data["messages"][0]["content"]
        user_prompt = benchmark_data["messages"][1]["content"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The description is {user_prompt}"},
        ]

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        end_time = time.time()

        time.sleep(1)  # Sleep for 1 second to avoid rate limiting

        return response.choices[0].message.content, end_time - start_time


class MistralInstructScorer(ModelScorer):
    def __init__(self, endpoint_url, model_id="base", model_type="mistral"):
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.model_type = model_type

    def process_with_model(self, benchmark_data):
        system_prompt = benchmark_data["messages"][0]["content"]
        user_prompt = benchmark_data["messages"][1]["content"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": 1024,
        }

        start_time = time.time()
        Logger.info(f"[*] Running Mistral Instruct model: {self.model_id}")
        try:
            response = requests.post(self.endpoint_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            model_response = response_data.get("response", "")
        except requests.RequestException as e:
            Logger.error(f"Request to Mistral Instruct API failed: {e}")
            return None, None

        end_time = time.time()
        Logger.info(f"[*] Response from Mistral Instruct: {model_response}")
        return model_response, end_time - start_time


class CustomModelScorer(ModelScorer):
    def __init__(
        self,
        endpoint_url,
        model_id="custom",
        model_type="custom",
        temperature=0.9,
        max_tokens=1024,
    ):
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens

    def process_with_model(self, benchmark_data):
        system_prompt = benchmark_data["messages"][0]["content"]
        user_prompt = benchmark_data["messages"][1]["content"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        start_time = time.time()
        Logger.info(f"[*] Running custom model: {self.model_id}")
        try:
            response = requests.post(self.endpoint_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            model_response = response_data.get("response", "")
        except requests.RequestException as e:
            Logger.error(f"Request to custom model API failed: {e}")
            return None, None

        end_time = time.time()
        Logger.info(f"[*] Response from custom model: {model_response}")
        return model_response, end_time - start_time


def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def main():
    # Load the configuration file
    config = load_config("config.yaml")

    # Setup logging based on configuration
    Logger.setup_logging(config.get("log_file", None))

    # Specify the testing file
    evaluation_file = config.get("testing_file", "evaluation/reconnaissance.jsonl")
    Logger.info("[*] Current Working Directory: %s", os.getcwd())

    # Check if the testing file exists
    if not os.path.exists(evaluation_file):
        Logger.error("[*] File does not exist or is not accessible.")
        return

    # Initialize the ScoreProcessor
    score_processor = ScoreProcessor()

    # Initialize and run model benchmarks
    scorers = [
        # OpenAIScorer(api_key=config.get("openai_api_key"), model_id="gpt-4-turbo"),
        # OpenAIScorer(api_key=config.get("openai_api_key"), model_id="gpt-4o-mini"),
        # OpenAIScorer(api_key=config.get("openai_api_key"), model_id="gpt-3.5-turbo"),
        # OpenAIScorer(api_key=config.get("openai_api_key"), model_id="gpt-4o"),
        MistralInstructScorer(
            endpoint_url=config.get("mistral_instruct_endpoint"),
            model_id="mistral-7ep",
        ),
    ]

    for scorer in scorers:
        score_processor.run_model_benchmarks(scorer, config, evaluation_file)
        score_processor.write_model_scores_to_csv(
            scorer.model_type, scorer.model_id, config.get("output_dir", "scores")
        )
        time.sleep(5)  # Sleep for 5 seconds between models avoids rate limiting


if __name__ == "__main__":
    main()
