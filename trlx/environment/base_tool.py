### Adapted from trl: https://github.com/huggingface/trl/blob/main/trl/environment/base_environment.py
import re


class ToolEnvironment:

    """
    LLM interaction with the tool to get the reward
    """

    def __init__(self, tools=None, prompt=None, reward_fn=None):
        if isinstance(tools, dict):
            self.tools = tools
        else:
            self.tools = dict([(tool.__class__.__name__, tool) for tool in tools])
        self.prompt = prompt
        self.reward_fn = reward_fn
        self.request_token = "<request>"
        self.call_token = "<call>"
        self.response_token = "<response>"
        self.submit_token = "<submit>"

    def parse_tool_call(self, text):
        """
        Parse request string. Expected format: <request><tool_name>query<call>
        """
        result = re.search(f"(?<={self.request_token}).*?(?={self.call_token})", text, re.DOTALL)

        # if we can't find a <request>/<call> span we return none
        if result is None:
            return None, None
        else:
            extracted_text = result.group()

        result = re.search(r"<(.*?)>", extracted_text)

        # if we can't find a tool name we return none
        if result is None:
            return None, None
        else:
            tool = result.group(1)

        # split off the tool name
        query = ">".join(extracted_text.split(">")[1:])

        return tool, query

    def get_reward(self, texts, **kwargs):
        """
        Get the reward for the generated text
        """
        tool_responses = [self.execution(text) for text in texts]
        reward = self.reward_fn(tool_responses, **kwargs)
        return reward

    def _get_generated_text(self, text):
        text = text.strip()
        text = text.replace("<|endoftext|>", "")
        text = text.replace("</s>", "")
        if not text.endswith(self.call_token):
            text = f"{text}{self.call_token}"
        return text[len(self.prompt) :]

    def execution(self, text):
        """
        Tool execution and get reward
        """
        generated_text = self._get_generated_text(text)
        tool, query = self.parse_tool_call(generated_text)
        if tool is None or query is None:
            response = f"Unknown tool call: {query}"
        else:
            if tool not in self.tools:
                response = f"Unknown tool {tool}."
            try:
                response = self.tools[tool](query)
            except Exception as error:
                response = f"Tool Error: {str(error)}"
        return response
