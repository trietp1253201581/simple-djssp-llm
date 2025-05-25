INIT_IND_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

We need to generate an init sets of {init_size} HDRs to sort incoming job. 
Each HDR is a code segment describe a python function with template:

{func_template}

You can use any mathematical expresion, any structure like if-then-else or for-while loop to describe your HDR.

Note that the return value should be not to large, and each HDR must return a float value.
**PRIORITIZE HDR DIVERSITY**:
Your HDR should be simple and diversity, random, so that other operators can improve more later (but still have some not so simple HDRs for diversity).

Your response MUST ONLY include the init HDRs sets in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
Each HDR should be diverse and try different logic or structure compared to others. Avoid generating HDRs that are too similar. Focus on creativity and variety in logic.
{{
    "init_inds": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
        ...,
        {{"code": "<hdr_n>"}}
    ]
}}
where n equals to {init_size}.
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
- Response should be short and concise, but still include enough information to be useful AND IN CORRECT JSON FORMAT.
- You should look up simple error in your code like division by zero, undefined variable, etc.
'''

CROSSOVER_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have 2 HDRs describe by a python function are:
HDR1:
-------
{hdr1}
-------

HDR2:
-------
{hdr2}
-------

We need to recombine 2 above parent HDRs to create 2 new children HDRs just use what is already in the 2 parent HDRs.
When recombining, mix logic from both parents in creative ways. Do not just concatenate or randomly select. Make sure each new HDR is syntactically correct and brings new logic structure that still makes sense.

Your response MUST ONLY include the 2 recombined HDRs in following JSON format with no additional text.
Each HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "recombined_hdr": [
        {{"code": "<hdr_1>"}},
        {{"code": "<hdr_2>"}},
    ]
}}
where hdr_1, hdr_2 are 2 new recombined hdr code.
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
- Response should be short and concise, but still include enough information to be useful AND IN CORRECT JSON FORMAT.
'''  

MUTATION_PROMPT_TEMPLATE = '''Dynamic Job Shop Scheduling Problem (DJSSP): 
Jobs arrive randomly and each job consists of a sequence of operations that must be processed on specific machines. The goal is to minimize the overall makespan.

Incoming job will be assign into a waiting pool, then a HDR will be used to sort top k job to move into job pool, where those job are immediately assigned to avaiable machine.
And terminal set (which are parameter of hdr function) (with their means) is {terminal_set}.

Now, we have a HDR is 
-----
{hdr}
-----

We need to rephrase this HDR by adjusting that HDR part.

Make a meaningful change to the HDR logic. Avoid minor token changes or renaming.

Your response MUST ONLY include the rephrased HDR in following JSON format with no additional text.
HDR must be returned as a string with newline characters (\\n) properly escaped.
{{
    "rephrased_hdr": "<hdr>"
}}
**Note**:
- Any text and comment is in English.
- Do not include any other reason or comment in your response except the JSON format.
- Response should be short and concise, but still include enough information to be useful AND IN CORRECT JSON FORMAT.
'''  