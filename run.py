import json

from solution import Solution

if __name__ == '__main__':
    s = Solution()
    json_lines = []

    with open("tpc-dataset.train.txt", 'r') as f:
        for line in f:
            json_lines.append(json.loads(line))
    # s.train(json_lines)
    print(s.get_age(["hi", "ejwj"]))
