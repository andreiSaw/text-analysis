import json

from solution import Solution

if __name__ == '__main__':
    s = Solution()
    json_lines = []

    with open("tpc-dataset.train.txt", 'r') as f:
        for line in f:
            json_lines.append(json.loads(line))
    print("opened1\n")
    s.train(json_lines)
    print("trained2\n")
    print(s.get_education(["hi", "ejwj"]))
