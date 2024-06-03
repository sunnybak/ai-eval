

from openai import OpenAI

client = OpenAI()


def meme_gen(enable_user_funnier, prompt, model):
    meme = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a meme generator."},
            {"role": "user", "content": "Create a meme"},
        ],
    )
    return meme.choices[0].message.content

# make sure meme is funny
# see if GPT-4 makes better memes


from ai_eval.scorers.nlp.string_scorers import StringScorer
from ai_eval.evaluator import Evaluator

score = StringScorer(meme).topics(['joke', 'no joke', ''], threshold=0.8)

evals = Evaluator(target=['joke'], scorer=score)
# consistency

# string_scorers.funny()

print(score)


def test_meme():
    
    assert evals(meme_gen())


from ai_eval import run_experiment

# input = constraints(resources, team, capital, ICP)
# generate ideas
# create a vision and mission
# list all assumptions for business model
# come up with a list of tasks to test the riskiest assumptions
# output = top 3 ideas with SWOT analysis

# build a generator for inputs with weights for criteria
# write basic prompts for chain of thought + evals/scorers (LLM-based scores + aggregation formula)
# define the scorers
# run the experiment

scores_df = run_experiment(
    app=meme_gen,
    args=(enable_user_funnier,),
    hyperparam_dict={
        'model': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-turbo'],
        'prompt': ["Make it funnier", "Be funny."],
    },
    consistency=4
)

# product idea generator
# scores:
# - meme potential
# - relevance
# - disruptiveness
# - AI
# - rags to riches




# define the target/scorer/eval
# run the experiment/simulation
# iterate
# test 