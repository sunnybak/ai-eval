from ai_eval.scorers.AbsBaseScorer import AbsBaseScorer
from ai_eval.scorers.score import scorer
from ai_eval.scorers.utils import gemini_call
import yaml

class LLMScorer(AbsBaseScorer):
    def __init__(self, provider, model):
        self.model = model
        self.provider = provider
        self.provider_auth = {
            "api_key": None,
        }

    def _get_eval_prompt(self):
        # read the yaml file prompts/g-eval.yaml
        # get the system_prompt & user_prompt
        yaml_file = "prompts/g-eval.yaml"
        with open(yaml_file, "r") as file:
            prompts = yaml.load(file, Loader=yaml.FullLoader)
        return prompts["system_prompt"], prompts["user_prompt"]

    def _predict(self, prompt):
        if self.provider == "openai":
            return openai_call(
                prompt=prompt,
                model=self.model,
            )
        elif self.provider == "google":
            return gemini_call(
                prompt=prompt,
                model=self.model,
            )
        else:
            raise ValueError("Invalid provider")

    def _parse_response(self, output):
        explanation_start = output.index('Rating:')
        if explanation_start != -1:
            explanation = output[:explanation_start]
        else:
            explanation = 'parsing error'
        rating_start = output.index('Rating: [[')
        rating_end = output.index(']]', rating_start)
        if rating_start != -1:
            rating = float(output[rating_start + 10:rating_end])
        else:
            rating = -1
        return (rating, explanation)

    @scorer
    def score(
        self,
        criteria,
        inputs,
        outputs,
        context=None,
        ground_truth=None,
        **kwargs,
    ):
        system_prompt, user_prompt = self._get_eval_prompt()
        # replace {{ criteria }} with the actual criteria 
        system_prompt = system_prompt.replace("{{ criteria }}", criteria)

        if type(inputs) != str:
            inputs = str(inputs)

        # replace {{ inputs }} with the actual inputs
        user_prompt = user_prompt.replace("{{ inputs }}", inputs)

        if type(outputs) != str:
            outputs = str(outputs)

        # replace {{ outputs }} with the actual outputs
        user_prompt = user_prompt.replace("{{ outputs }}", outputs)

        # if context / ground_truth are provided, append those
        if context:
            if type(context) != str:
                context = str(context)
            user_prompt = user_prompt + "\n[Context]\n" + context
        if ground_truth:
            if type(ground_truth) != str:
                ground_truth = str(ground_truth)
            user_prompt = user_prompt + "\n[Ground Truth]\n" + ground_truth
        # check if any kwargs are provided
        if kwargs:
            # for each kwarg, do as we did with context and ground_truth
            for key, value in kwargs.items():
                if type(value) != str:
                    value = str(value)
                user_prompt = user_prompt + f"\n[{key}]\n" + value

        prompt = system_prompt + "\n" + user_prompt
        print("calling predict with prompt: ", prompt)
        print("provider: ", self.provider)
        response = self._predict(prompt)
        return self._parse_response(response)

if __name__ == "__main__":
    llm = LLMScorer(provider="google", model="gemini-1.5-pro-latest")
    print(llm.score(
        criteria="funniest output for the input",
        inputs="What do you call a bear with no teeth?",
        outputs="A gummy bear",
    ))
