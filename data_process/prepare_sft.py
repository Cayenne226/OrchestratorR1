"""
Generate SFT warmup examples covering all 6 Agent types.

Each example shows the model the expected <call>/<answer> tag format.
These are hand-crafted to ensure the model can produce valid tags
before GRPO training begins.

Usage:
    python data_process/prepare_sft.py --output data/sft_warmup.jsonl
"""

import argparse
import json
from pathlib import Path


# Warmup examples covering all agent types and complexity levels
SFT_EXAMPLES = [
    # === Simple: executor_cheap only ===
    {
        "input": "When did World War 2 end?",
        "output": '<think>This is a simple factual question. I can answer it directly with a cheap executor.</think>\n<call type="executor_cheap">When did World War 2 end?</call>\n<answer>World War 2 ended on September 2, 1945.</answer>',
    },
    {
        "input": "What is the capital of France?",
        "output": '<think>Simple factual question, direct lookup.</think>\n<call type="executor_cheap">What is the capital of France?</call>\n<answer>Paris</answer>',
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "output": '<think>Simple factual question about literature.</think>\n<call type="executor_cheap">Who wrote Romeo and Juliet?</call>\n<answer>William Shakespeare</answer>',
    },
    {
        "input": "What is the chemical formula for water?",
        "output": '<think>Basic chemistry fact, use cheap executor.</think>\n<call type="executor_cheap">What is the chemical formula for water?</call>\n<answer>H2O</answer>',
    },
    {
        "input": "How many continents are there?",
        "output": '<think>Simple geography question.</think>\n<call type="executor_cheap">How many continents are there?</call>\n<answer>7</answer>',
    },
    {
        "input": "What year was the Declaration of Independence signed?",
        "output": '<think>Historical fact, straightforward.</think>\n<call type="executor_cheap">What year was the Declaration of Independence signed?</call>\n<answer>1776</answer>',
    },
    {
        "input": "What is the largest planet in our solar system?",
        "output": '<think>Basic astronomy fact.</think>\n<call type="executor_cheap">What is the largest planet in our solar system?</call>\n<answer>Jupiter</answer>',
    },
    {
        "input": "Who painted the Mona Lisa?",
        "output": '<think>Simple art history question.</think>\n<call type="executor_cheap">Who painted the Mona Lisa?</call>\n<answer>Leonardo da Vinci</answer>',
    },

    # === Refiner: vague input ===
    {
        "input": "write code",
        "output": '<think>The input is extremely vague. I need to refine it before executing.</think>\n<call type="refiner">write code</call>\n<call type="executor_cheap">Write a simple Python function that takes a list of numbers and returns their sum</call>\n<answer>def sum_list(numbers):\n    return sum(numbers)</answer>',
    },
    {
        "input": "help me with my project",
        "output": '<think>Very vague request. Need to clarify what the user wants.</think>\n<call type="refiner">help me with my project</call>\n<call type="executor_cheap">Provide a general project planning template with phases: planning, development, testing, and deployment</call>\n<answer>A typical project follows these phases:\n1. Planning: Define goals, scope, and timeline\n2. Development: Implement core features\n3. Testing: Verify functionality and fix bugs\n4. Deployment: Release and monitor</answer>',
    },
    {
        "input": "make it better",
        "output": '<think>Extremely vague. Must refine to understand what needs improvement.</think>\n<call type="refiner">make it better</call>\n<call type="executor_cheap">List common ways to improve code quality: add error handling, write tests, improve naming, add documentation, optimize performance</call>\n<answer>Common ways to improve code:\n1. Add error handling\n2. Write unit tests\n3. Use descriptive variable names\n4. Add documentation\n5. Optimize performance bottlenecks</answer>',
    },
    {
        "input": "explain that thing from yesterday",
        "output": '<think>Unclear reference. Need to refine and make a reasonable interpretation.</think>\n<call type="refiner">explain that thing from yesterday</call>\n<call type="executor_cheap">Without specific context, explain general strategies for recalling and understanding recent topics: review notes, check recent communications, and identify key concepts</call>\n<answer>I don\'t have context about what was discussed yesterday. To help you, please provide more details about the specific topic you\'d like explained.</answer>',
    },

    # === Decomposer: complex multi-step tasks ===
    {
        "input": "Compare the GDP of Japan and Germany in 2023",
        "output": '<think>This requires looking up two separate facts and then comparing. I should decompose it.</think>\n<call type="decomposer">Compare the GDP of Japan and Germany in 2023</call>\n<call type="executor_cheap">What was Japan\'s GDP in 2023?</call>\n<call type="executor_cheap">What was Germany\'s GDP in 2023?</call>\n<call type="synthesizer">Compare Japan GDP and Germany GDP based on the data retrieved</call>\n<answer>In 2023, Germany\'s GDP was approximately $4.46 trillion, while Japan\'s GDP was approximately $4.23 trillion. Germany slightly surpassed Japan to become the world\'s third-largest economy.</answer>',
    },
    {
        "input": "What is the relationship between the author of Harry Potter and the university where she studied?",
        "output": '<think>Multi-hop question: need to find the author, then find where she studied.</think>\n<call type="decomposer">Find the relationship between the Harry Potter author and her university</call>\n<call type="executor_cheap">Who is the author of Harry Potter?</call>\n<call type="executor_cheap">Where did J.K. Rowling study at university?</call>\n<call type="synthesizer">Combine information about J.K. Rowling and her university education</call>\n<answer>J.K. Rowling, the author of Harry Potter, studied at the University of Exeter where she earned a degree in French and Classics.</answer>',
    },
    {
        "input": "Which country has a larger population, the birthplace of Mozart or the birthplace of Beethoven?",
        "output": '<think>Multi-hop: find birthplaces, then compare populations. Need decomposition.</think>\n<call type="decomposer">Compare populations of Mozart\'s and Beethoven\'s birth countries</call>\n<call type="executor_cheap">Where was Mozart born?</call>\n<call type="executor_cheap">Where was Beethoven born?</call>\n<call type="executor_cheap">What are the current populations of Austria and Germany?</call>\n<answer>Beethoven was born in Bonn, Germany, and Mozart was born in Salzburg, Austria. Germany has a larger population (approximately 84 million) compared to Austria (approximately 9 million).</answer>',
    },

    # === Executor_strong: complex reasoning ===
    {
        "input": "Implement a binary search tree with insert, delete, and search operations in Python",
        "output": '<think>This is a complex coding task requiring careful implementation. Use the strong executor.</think>\n<call type="executor_strong">Implement a binary search tree in Python with insert, delete, and search operations. Include a Node class and BST class with proper handling of all delete cases (leaf, one child, two children).</call>\n<answer>class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n\nclass BST:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, val):\n        self.root = self._insert(self.root, val)\n\n    def _insert(self, node, val):\n        if not node:\n            return Node(val)\n        if val < node.val:\n            node.left = self._insert(node.left, val)\n        elif val > node.val:\n            node.right = self._insert(node.right, val)\n        return node\n\n    def search(self, val):\n        return self._search(self.root, val)\n\n    def _search(self, node, val):\n        if not node or node.val == val:\n            return node\n        if val < node.val:\n            return self._search(node.left, val)\n        return self._search(node.right, val)\n\n    def delete(self, val):\n        self.root = self._delete(self.root, val)\n\n    def _delete(self, node, val):\n        if not node:\n            return None\n        if val < node.val:\n            node.left = self._delete(node.left, val)\n        elif val > node.val:\n            node.right = self._delete(node.right, val)\n        else:\n            if not node.left:\n                return node.right\n            if not node.right:\n                return node.left\n            successor = node.right\n            while successor.left:\n                successor = successor.left\n            node.val = successor.val\n            node.right = self._delete(node.right, successor.val)\n        return node</answer>',
    },
    {
        "input": "Explain the difference between TCP and UDP with examples of when to use each",
        "output": '<think>This requires detailed technical explanation. Use strong executor for accuracy.</think>\n<call type="executor_strong">Explain TCP vs UDP protocols, their key differences, and provide real-world use cases for each</call>\n<answer>TCP (Transmission Control Protocol) is connection-oriented, guarantees delivery order, and provides error checking. Use TCP for: web browsing (HTTP), email (SMTP), file transfers (FTP).\n\nUDP (User Datagram Protocol) is connectionless, faster but unreliable, with no delivery guarantee. Use UDP for: video streaming, online gaming, DNS lookups, VoIP.\n\nKey differences:\n- Reliability: TCP guarantees delivery; UDP does not\n- Speed: UDP is faster due to less overhead\n- Connection: TCP requires handshake; UDP is fire-and-forget\n- Ordering: TCP maintains order; UDP may deliver out of order</answer>',
    },

    # === Critic: verify results ===
    {
        "input": "Write a function to check if a number is prime and verify it works correctly",
        "output": '<think>Need to write code and verify correctness. Use executor then critic.</think>\n<call type="executor_cheap">Write a Python function is_prime(n) that returns True if n is prime, False otherwise</call>\n<call type="critic">Verify the is_prime function: does it handle edge cases (0, 1, 2, negative numbers)? Is the algorithm correct?</call>\n<answer>def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True\n\n# Handles all edge cases: negatives, 0, 1 return False; 2 returns True; even numbers quickly eliminated.</answer>',
    },
    {
        "input": "Solve: If a train travels 120km in 2 hours, and then 180km in 3 hours, what is its average speed?",
        "output": '<think>Math problem. Solve it and verify the calculation.</think>\n<call type="executor_cheap">Calculate: total distance = 120 + 180 = 300km, total time = 2 + 3 = 5 hours, average speed = 300/5</call>\n<call type="critic">Verify: is the average speed calculation correct? Average speed = total distance / total time, not the average of individual speeds.</call>\n<answer>The average speed is 60 km/h. Total distance = 120 + 180 = 300 km. Total time = 2 + 3 = 5 hours. Average speed = 300/5 = 60 km/h.</answer>',
    },

    # === Synthesizer: combining multiple results ===
    {
        "input": "Give me a summary of the three largest oceans by area",
        "output": '<think>Need to look up info about three oceans, then combine into a summary.</think>\n<call type="executor_cheap">What is the area and key facts about the Pacific Ocean?</call>\n<call type="executor_cheap">What is the area and key facts about the Atlantic Ocean?</call>\n<call type="executor_cheap">What is the area and key facts about the Indian Ocean?</call>\n<call type="synthesizer">Combine the facts about Pacific, Atlantic, and Indian oceans into a concise summary</call>\n<answer>The three largest oceans:\n1. Pacific Ocean: ~165.25 million km², the largest and deepest ocean\n2. Atlantic Ocean: ~106.46 million km², separates the Americas from Europe/Africa\n3. Indian Ocean: ~70.56 million km², the warmest ocean</answer>',
    },

    # === Full pipeline: refiner + decomposer + executor + critic + synthesizer ===
    {
        "input": "build api with auth",
        "output": '<think>Vague input about building an API. Need to refine first, then decompose the complex task.</think>\n<call type="refiner">build api with auth</call>\n<call type="decomposer">Build a REST API with JWT authentication in Python using FastAPI, including user registration, login, and protected endpoints</call>\n<call type="executor_strong">Implement JWT authentication utilities and user registration/login endpoints in FastAPI</call>\n<call type="executor_cheap">Add a protected endpoint example that requires valid JWT token</call>\n<call type="critic">Review the API implementation: does it handle token expiration, password hashing, and error responses properly?</call>\n<call type="synthesizer">Combine all API components into a complete working FastAPI application</call>\n<answer>Complete FastAPI application with JWT auth:\n\n1. JWT utilities (create/verify tokens)\n2. User registration with password hashing\n3. Login endpoint returning JWT\n4. Protected endpoints with token verification\n5. Proper error handling for expired/invalid tokens</answer>',
    },

    # === More simple QA for balance ===
    {
        "input": "What is the speed of light?",
        "output": '<think>Simple physics fact.</think>\n<call type="executor_cheap">What is the speed of light?</call>\n<answer>The speed of light in vacuum is approximately 299,792,458 meters per second (about 3 × 10^8 m/s).</answer>',
    },
    {
        "input": "Who is the current president of the United States?",
        "output": '<think>Simple factual question about current affairs.</think>\n<call type="executor_cheap">Who is the current president of the United States?</call>\n<answer>Joe Biden is the 46th president of the United States.</answer>',
    },
    {
        "input": "What is photosynthesis?",
        "output": '<think>Science concept explanation, straightforward.</think>\n<call type="executor_cheap">What is photosynthesis?</call>\n<answer>Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food from carbon dioxide and water. It produces oxygen as a byproduct.</answer>',
    },
    {
        "input": "Convert 100 Fahrenheit to Celsius",
        "output": '<think>Simple conversion calculation.</think>\n<call type="executor_cheap">Convert 100°F to Celsius using formula C = (F - 32) × 5/9</call>\n<answer>100°F = 37.78°C</answer>',
    },

    # === More multi-hop ===
    {
        "input": "Was the director of Inception born before or after the director of The Matrix?",
        "output": '<think>Multi-hop: need to find both directors and their birth years.</think>\n<call type="decomposer">Compare birth years of Inception and Matrix directors</call>\n<call type="executor_cheap">Who directed Inception and when was he born?</call>\n<call type="executor_cheap">Who directed The Matrix and when were they born?</call>\n<answer>Christopher Nolan (Inception) was born on July 30, 1970. The Wachowskis (The Matrix) - Lilly was born on December 29, 1967, and Lana on June 21, 1965. So the Inception director was born after the Matrix directors.</answer>',
    },

    # === Code with critic ===
    {
        "input": "Write a function to find the longest common subsequence of two strings",
        "output": '<think>Classic dynamic programming problem. Needs careful implementation and verification.</think>\n<call type="executor_strong">Implement longest common subsequence (LCS) using dynamic programming in Python. Return both the length and the actual subsequence string.</call>\n<call type="critic">Verify the LCS implementation: check time complexity is O(mn), space is O(mn), handles empty strings, and correctly backtracks to find the subsequence.</call>\n<answer>def lcs(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[""] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + s1[i-1]\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)\n    return dp[m][n]\n\n# Time: O(mn), Space: O(mn)\n# Example: lcs("ABCBDAB", "BDCAB") -> "BCAB"</answer>',
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/sft_warmup.jsonl")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in SFT_EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Print statistics
    agent_types = {}
    for ex in SFT_EXAMPLES:
        import re
        for match in re.finditer(r'<call type="(\w+)"', ex["output"]):
            t = match.group(1)
            agent_types[t] = agent_types.get(t, 0) + 1

    print(f"Generated {len(SFT_EXAMPLES)} SFT warmup examples → {args.output}")
    print("Agent type coverage:")
    for t, c in sorted(agent_types.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
