import os
from crewai import Agent, Crew, Process, Task
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage
from typing import List, Optional, Iterator, Dict, Any
import requests
import json
import yaml


class ChatOllamaNoThink(ChatOllama):
    """ChatOllama that disables Qwen3.5 thinking mode for cleaner agent output."""

    think: bool = False

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        if "options" in kwargs:
            params["options"] = kwargs["options"]
        else:
            params["options"] = {
                **params["options"],
                "stop": stop,
                **{k: v for k, v in kwargs.items() if k not in self._default_params},
            }

        if payload.get("messages"):
            request_payload = {"messages": payload.get("messages", []), **params}
        else:
            request_payload = {
                "prompt": payload.get("prompt"),
                "images": payload.get("images", []),
                **params,
            }

        # Inject think parameter at top level for Ollama API
        if self.think is not None:
            request_payload["think"] = self.think

        response = requests.post(
            url=api_url,
            headers={
                "Content-Type": "application/json",
                **(self.headers if isinstance(self.headers, dict) else {}),
            },
            json=request_payload,
            stream=True,
            timeout=self.timeout,
        )
        response.encoding = "utf-8"
        if response.status_code != 200:
            if response.status_code == 404:
                from langchain_community.chat_models.ollama import OllamaEndpointNotFoundError
                raise OllamaEndpointNotFoundError(
                    "Ollama call failed with status code 404. "
                    "Maybe your model is not found "
                    f"and you should pull the model with `ollama pull {self.model}`."
                )
            else:
                optional_detail = response.json().get("error")
                raise ValueError(
                    f"Ollama call failed with status code {response.status_code}."
                    f" Details: {optional_detail}"
                )
        return response.iter_lines(decode_unicode=True)


class MarketingCrew:
    def __init__(self):
        # Load configs
        with open('config/agents.yaml', 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open('config/tasks.yaml', 'r') as f:
            self.tasks_config = yaml.safe_load(f)

        # Use local Ollama LLM instead of OpenAI
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

        self.llm = ChatOllamaNoThink(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7,
            think=False,
        )

    def create_agents(self):
        self.researcher = Agent(
            role=self.agents_config['market_researcher']['role'],
            goal=self.agents_config['market_researcher']['goal'],
            backstory=self.agents_config['market_researcher']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        self.strategist = Agent(
            role=self.agents_config['content_strategist']['role'],
            goal=self.agents_config['content_strategist']['goal'],
            backstory=self.agents_config['content_strategist']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        self.visual_creator = Agent(
            role=self.agents_config['visual_creator']['role'],
            goal=self.agents_config['visual_creator']['goal'],
            backstory=self.agents_config['visual_creator']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        self.copywriter = Agent(
            role=self.agents_config['copywriter']['role'],
            goal=self.agents_config['copywriter']['goal'],
            backstory=self.agents_config['copywriter']['backstory'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def create_tasks(self):
        self.task_research = Task(
            description=self.tasks_config['market_research']['description'],
            expected_output=self.tasks_config['market_research']['expected_output'],
            agent=self.researcher
        )

        self.task_strategy = Task(
            description=self.tasks_config['content_strategy']['description'],
            expected_output=self.tasks_config['content_strategy']['expected_output'],
            agent=self.strategist
        )

        self.task_visuals = Task(
            description=self.tasks_config['visual_content_creation']['description'],
            expected_output=self.tasks_config['visual_content_creation']['expected_output'],
            agent=self.visual_creator
        )

        self.task_copy = Task(
            description=self.tasks_config['copywriting']['description'],
            expected_output=self.tasks_config['copywriting']['expected_output'],
            agent=self.copywriter
        )

        self.task_report = Task(
            description=self.tasks_config['report_final_content_strategy']['description'],
            expected_output=self.tasks_config['report_final_content_strategy']['expected_output'],
            agent=self.strategist,
        )

    def run(self):
        self.create_agents()
        self.create_tasks()

        crew = Crew(
            agents=[self.researcher, self.strategist, self.visual_creator, self.copywriter],
            tasks=[self.task_research, self.task_strategy, self.task_visuals, self.task_copy, self.task_report],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        # Save the final output to file (crewai 0.5.0 doesn't support output_file param)
        with open("final-content-strategy.md", "w") as f:
            f.write(str(result))
        print("Final strategy saved to 'final-content-strategy.md'")

        return result
