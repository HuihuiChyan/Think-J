cot_judge_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response (a)]
{response1}
[The End of Response (a)]

[The Start of Response (b)]
{response2}
[The End of Response (b)]

Please provide an evaluation by first offering a detailed explanation, and then end your answer with "Therefore, Response (a) is better." or "Therefore, Response (b) is better."."""}

cot_critic_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response (a)]
{response1}
[The End of Response (a)]

[The Start of Response (b)]
{response2}
[The End of Response (b)]

Given that {chosen} is better than {rejected}, please provide an evaluation by first offering a detailed explanation, and then end your answer with "Therefore, Response (a) is better." or "Therefore, Response (b) is better."."""}

strength_judge_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response (a)]
{response1}
[The End of Response (a)]

[The Start of Response (b)]
{response2}
[The End of Response (b)]

Please provide an evaluation by first offering a detailed explanation. Then end your answer with the following format: 'Therefore, Response (a/b) is better, and the strength is [[score]].'. For example: 'Therefore, Response (b) is better, and the strength is [[1]].'.
The strength denotes how much one response is preferred over the other, with the following scale:
- A strength of 1 indicates one response is slightly better than the other.
- A strength of 2 indicates one response is better than the other.
- A strength of 3 indicates one response is much better than the other."""}

strength_critic_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response (a)]
{response1}
[The End of Response (a)]

[The Start of Response (b)]
{response2}
[The End of Response (b)]

Given that {chosen} is better than {rejected}, please provide an evaluation by first offering a detailed explanation. Then end your answer with the following format: 'Therefore, Response (a/b) is better, and the strength is [[score]].'. For example: 'Therefore, Response (b) is better, and the strength is [[1]].'.
The strength denotes how much one response is preferred over the other, with the following scale:
- A strength of 1 indicates one response is slightly better than the other.
- A strength of 2 indicates one response is better than the other.
- A strength of 3 indicates one response is much better than the other."""}

single_score_judge_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response]
{response}
[The End of Response]

Please provide your evaluation in the following format:
- Begin with a detailed explanation of the response's strengths and weaknesses.
- Conclude with "Therefore, the score for the response is <Score>score</Score>.", for example, "Therefore, the score the response is <Score>80</Score>.".
- A score of 0 indicates that the response is completely incorrect, unclear, or irrelevant.
- A score of 100 indicates that the response is perfectly accurate, clear, relevant, and demonstrates exemplary quality."""}

double_score_judge_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response 1]
{response1}
[The End of Response 1]

[The Start of Response 2]
{response2}
[The End of Response 2]

Please provide your evaluation in the following format:
- Begin with a detailed explanation of the strengths and weaknesses of the responses.
- Conclude with two scores indicating the quality of the two responses, formatted as <Score 1>score1</Score 1> and <Score 2>score2</Score 2>. 
- For example, you can conclude with: "Therefore, the scores for the responses are <Score 1>30</Score 1> and <Score 2>80</Score 2>, respectively.".
- A score of 0 indicates that the response is completely incorrect, unclear, or irrelevant.
- A score of 100 indicates that the response is perfectly accurate, clear, relevant, and demonstrates exemplary quality."""}

double_score_critic_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response 1]
{response1}
[The End of Response 1]

[The Start of Response 2]
{response2}
[The End of Response 2]

Given that {chosen} is better than {rejected}, please provide your evaluation in the following format:
- Begin with a detailed explanation of the strengths and weaknesses of the responses.
- Conclude with two scores indicating the quality of the two responses, formatted as <Score 1>score1</Score 1> and <Score 2>score2</Score 2>. 
- For example, you can conclude with: "Therefore, the scores for the responses are <Score 1>30</Score 1> and <Score 2>80</Score 2>, respectively.".
- A score of 0 indicates that the response is completely incorrect, unclear, or irrelevant.
- A score of 100 indicates that the response is perfectly accurate, clear, relevant, and demonstrates exemplary quality."""}

direct_judge_prompt = {
    "system": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
    "user": """[The Start of User Question]
{instruction}
[The End of User Question]

[The Start of Response (a)]
{response1}
[The End of Response (a)]

[The Start of Response (b)]
{response2}
[The End of Response (b)]

Please provide an evaluation by directly answer with "Response (a) is better." or "Response (b) is better.". Do not generate any other openings, closings or explanations."""}