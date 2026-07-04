# AI Coding Assignment

# Part 1:
# Page 1: DFS, BFS, UCS ve A* Algoritmalarının Karşılaştırılması
# Page 2: 6×6 Sudoku Çözümü (Backtracking)
# Page 3: Monte Carlo Tree Search ile Tic-Tac-Toe
# Page 4: Policy Iteration ile 3×4 GridWorld MDP Çözümü

# Part 2:
# Page 5: 4x4 GridWorld ortamında, rastgele yerleştirilen 2 engel ve 1 kazanma terminal durumuyla, Q-Öğrenme ajanı uygulanması.
# Page 6: Belirli bir veri kümesi üzerinde sıfırdan Naive Bayes sınıflandırıcı uygulanması.

# --- Part 2 - Page 5: Q-Learning Agent for 4x4 GridWorld ---

# --- Part 3 - Task 1: Custom Decision Tree Classifier ---

# --- Part 3 - Task 2: Single Hidden Layer Perceptron with Sigmoid ---


# --- Page 1: DFS, BFS, UCS, A* Comparison ---
import time
import heapq
from collections import deque

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

def dfs(start, goal, graph):
    stack = [Node(start)]
    visited = set()
    while stack:
        node = stack.pop()
        if node.state in visited:
            continue
        visited.add(node.state)
        if node.state == goal:
            return reconstruct_path(node)
        for neighbor in graph[node.state]:
            stack.append(Node(neighbor, node))
    return None

def bfs(start, goal, graph):
    queue = deque([Node(start)])
    visited = set()
    while queue:
        node = queue.popleft()
        if node.state in visited:
            continue
        visited.add(node.state)
        if node.state == goal:
            return reconstruct_path(node)
        for neighbor in graph[node.state]:
            queue.append(Node(neighbor, node))
    return None

def ucs(start, goal, graph, cost_fn):
    pq = [(0, Node(start))]
    visited = set()
    while pq:
        _, node = heapq.heappop(pq)
        if node.state in visited:
            continue
        visited.add(node.state)
        if node.state == goal:
            return reconstruct_path(node)
        for neighbor in graph[node.state]:
            heapq.heappush(pq, (node.cost + cost_fn(node.state, neighbor),
                                Node(neighbor, node, node.cost + cost_fn(node.state, neighbor))))
    return None

def astar(start, goal, graph, cost_fn, heuristic_fn):
    pq = [Node(start, heuristic=heuristic_fn(start, goal))]
    visited = set()
    while pq:
        node = heapq.heappop(pq)
        if node.state in visited:
            continue
        visited.add(node.state)
        if node.state == goal:
            return reconstruct_path(node)
        for neighbor in graph[node.state]:
            new_cost = node.cost + cost_fn(node.state, neighbor)
            heuristic = heuristic_fn(neighbor, goal)
            heapq.heappush(pq, Node(neighbor, node, new_cost, heuristic))
    return None

# --- Page 2: CSP Solver for 6x6 Sudoku (Backtracking) ---
SIZE = 6
BOX_HEIGHT = 2
BOX_WIDTH = 3

sudoku = [
    [0, 0, 0, 0, 5, 6],
    [0, 0, 0, 1, 0, 0],
    [0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0],
    [0, 0, 2, 0, 0, 0],
    [1, 5, 0, 0, 0, 0]
]

def is_valid(board, row, col, num):
    for i in range(SIZE):
        if board[row][i] == num or board[i][col] == num:
            return False
    box_row_start = (row // BOX_HEIGHT) * BOX_HEIGHT
    box_col_start = (col // BOX_WIDTH) * BOX_WIDTH
    for i in range(box_row_start, box_row_start + BOX_HEIGHT):
        for j in range(box_col_start, box_col_start + BOX_WIDTH):
            if board[i][j] == num:
                return False
    return True

def find_empty(board):
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] == 0:
                return i, j
    return None

def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True
    row, col = empty
    for num in range(1, SIZE + 1):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

# --- Page 3: MCTS Solver for 3x3 Tic-Tac-Toe ---
import math
import random

class TicTacToe:
    def __init__(self):
        self.board = [" "] * 9
        self.current_player = "X"

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, move):
        new_state = TicTacToe()
        new_state.board = self.board[:]
        new_state.board[move] = self.current_player
        new_state.current_player = "O" if self.current_player == "X" else "X"
        return new_state

    def is_terminal(self):
        return self.winner() is not None or " " not in self.board

    def winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a,b,c in wins:
            if self.board[a] == self.board[b] == self.board[c] != " ":
                return self.board[a]
        return None

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.available_moves()

    def ucb1(self):
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.make_move(move)
        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        if result == self.state.current_player:
            self.wins += 1

def mcts(root_state, iterations):
    root_node = MCTSNode(root_state)
    for _ in range(iterations):
        node = root_node
        while node.untried_moves == [] and node.children:
            node = node.best_child()
        if node.untried_moves:
            node = node.expand()
        state = node.state
        while not state.is_terminal():
            move = random.choice(state.available_moves())
            state = state.make_move(move)
        result = state.winner()
        while node is not None:
            node.update(result)
            node = node.parent
    return max(root_node.children, key=lambda c: c.visits).state

# --- Page 4: Policy Iteration for 3x4 GridWorld ---
import numpy as np

discount = 0.9
threshold = 1e-4
actions = ["U", "D", "L", "R"]
action_map = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

grid = [
    [0, 0, 0, 1],
    [0, None, 0, -1],
    [0, 0, 0, 0]
]

rows = len(grid)
cols = len(grid[0])
states = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] is not None]
policy = {s: random.choice(actions) for s in states if grid[s[0]][s[1]] == 0}
utilities = {s: 0 for s in states}

def is_terminal(s):
    val = grid[s[0]][s[1]]
    return val == 1 or val == -1

def get_next_state(s, a):
    dr, dc = action_map[a]
    r, c = s[0] + dr, s[1] + dc
    if 0 <= r < rows and 0 <= c < cols and grid[r][c] is not None:
        return (r, c)
    return s

def policy_evaluation():
    while True:
        delta = 0
        new_utilities = utilities.copy()
        for s in states:
            if is_terminal(s):
                continue
            a = policy[s]
            s_prime = get_next_state(s, a)
            reward = grid[s_prime[0]][s_prime[1]]
            new_utilities[s] = reward + discount * utilities[s_prime]
            delta = max(delta, abs(new_utilities[s] - utilities[s]))
        utilities.update(new_utilities)
        if delta < threshold:
            break

def policy_improvement():
    policy_stable = True
    for s in states:
        if is_terminal(s):
            continue
        old_action = policy[s]
        best_action = None
        best_value = float("-inf")
        for a in actions:
            s_prime = get_next_state(s, a)
            reward = grid[s_prime[0]][s_prime[1]]
            value = reward + discount * utilities[s_prime]
            if value > best_value:
                best_value = value
                best_action = a
        policy[s] = best_action
        if best_action != old_action:
            policy_stable = False
    return policy_stable

def policy_iteration():
    while True:
        policy_evaluation()
        if policy_improvement():
            break

# === PART 2 ===

# AI Coding Assignment
#
# Part 1:
# Sayfa 1: DFS, BFS, UCS ve A* algoritmalarının bir graf üzerinde uygulanması ve karşılaştırılması.
# Sayfa 2: 6×6 Sudoku çözümü için geri izlemeli (backtracking) kısıt tatmin algoritması.
# Sayfa 3: 3x3 Tic-Tac-Toe oyununda karar vermek için Monte Carlo Tree Search (MCTS) uygulanması.
# Sayfa 4: 3x4 GridWorld ortamında politika iterasyonu ile optimal MDP çözümünün elde edilmesi.
#
# Part 2:
# Sayfa 5: 4x4 GridWorld ortamında, rastgele yerleştirilen 2 engel ve 1 kazanma terminal durumuyla, Q-Öğrenme ajanı uygulanması.
# Sayfa 6: Belirli bir veri kümesi üzerinde sıfırdan Naive Bayes sınıflandırıcı uygulanması.

# --- Part 2 - Page 5: Q-Learning Agent for 4x4 GridWorld ---
import random

GRID_SIZE = 4
EPISODES = 500
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate
actions = ["U", "D", "L", "R"]
action_map = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def create_grid():
    from random import sample
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    win = sample([(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)], 1)[0]
    blocks = sample([(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r, c) != win], 2)
    grid[win[0]][win[1]] = 1
    for r, c in blocks:
        grid[r][c] = None
    return grid, win

def is_valid(state, grid):
    r, c = state
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and grid[r][c] is not None

def get_next_state(state, action, grid):
    dr, dc = action_map[action]
    next_state = (state[0] + dr, state[1] + dc)
    return next_state if is_valid(next_state, grid) else state

def q_learning(grid, win_state):
    q_table = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] is not None:
                q_table[(r, c)] = {a: 0 for a in actions}

    for episode in range(EPISODES):
        state = (random.randint(0, 3), random.randint(0, 3))
        while not is_valid(state, grid) or state == win_state:
            state = (random.randint(0, 3), random.randint(0, 3))

        while state != win_state:
            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_state = get_next_state(state, action, grid)
            reward = 1 if next_state == win_state else 0
            best_next = max(q_table[next_state].values()) if next_state in q_table else 0

            q_table[state][action] += ALPHA * (reward + GAMMA * best_next - q_table[state][action])
            state = next_state

    return q_table

random.seed(42)
grid, win_state = create_grid()
q_table = q_learning(grid, win_state)
print("Q-learning completed. Winning state:", win_state)

# --- Part 2 - Page 6: Naive Bayes Classifier from Scratch ---
from collections import defaultdict
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0

    def train(self, X, y):
        self.total_samples = len(y)
        for xi, label in zip(X, y):
            self.class_counts[label] += 1
            for idx, val in enumerate(xi):
                self.feature_counts[label][(idx, val)] += 1

    def predict(self, X):
        predictions = []
        for xi in X:
            class_scores = {}
            for c in self.class_counts:
                log_prob = math.log(self.class_counts[c] / self.total_samples)
                for idx, val in enumerate(xi):
                    feat = (idx, val)
                    feat_count = self.feature_counts[c][feat] + 1
                    total_feat = sum([self.feature_counts[c][(idx, v)] for v in set(v[idx] for v in X)]) + len(set(v[idx] for v in X))
                    log_prob += math.log(feat_count / total_feat)
                class_scores[c] = log_prob
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

X_train = [["sunny", "hot", "high"], ["sunny", "hot", "normal"], ["overcast", "cool", "normal"], ["rain", "mild", "high"]]
y_train = ["no", "yes", "yes", "no"]
X_test = [["sunny", "cool", "normal"], ["overcast", "hot", "high"]]

nb = NaiveBayesClassifier()
nb.train(X_train, y_train)
print("Predictions:", nb.predict(X_test))

# --- Part 3 - Task 1: Custom Decision Tree Classifier ---

import numpy as np

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini(y):
    classes = np.unique(y)
    return 1 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

def split_dataset(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = ~left_idx
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

def best_split(X, y):
    best_gini = float("inf")
    best_feat, best_thresh = None, None
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_l, y_l, X_r, y_r = split_dataset(X, y, feature, threshold)
            if len(y_l) == 0 or len(y_r) == 0:
                continue
            g = (len(y_l) / len(y)) * gini(y_l) + (len(y_r) / len(y)) * gini(y_r)
            if g < best_gini:
                best_gini, best_feat, best_thresh = g, feature, threshold
    return best_feat, best_thresh

def build_tree(X, y, use_split=True, depth=0, max_depth=3):
    if len(np.unique(y)) == 1 or depth == max_depth:
        value = np.bincount(y).argmax()
        return DecisionTreeNode(value=value)
    if not use_split:
        return DecisionTreeNode(value=np.bincount(y).argmax())
    feature, threshold = best_split(X, y)
    if feature is None:
        return DecisionTreeNode(value=np.bincount(y).argmax())
    X_l, y_l, X_r, y_r = split_dataset(X, y, feature, threshold)
    left = build_tree(X_l, y_l, use_split, depth + 1, max_depth)
    right = build_tree(X_r, y_r, use_split, depth + 1, max_depth)
    return DecisionTreeNode(feature, threshold, left, right)

def predict_tree(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

def evaluate_tree(X, y, use_split=True):
    tree = build_tree(X, y, use_split)
    predictions = np.array([predict_tree(tree, xi) for xi in X])
    accuracy = np.mean(predictions == y)
    return accuracy

# Test data
X = np.array([[2.7], [1.3], [3.1], [1.0], [2.0]])
y = np.array([1, 0, 1, 0, 0])
acc_with_split = evaluate_tree(X, y, use_split=True)
acc_without_split = evaluate_tree(X, y, use_split=False)

print("\nDecision Tree Results:")
print("Accuracy with splits:", acc_with_split)
print("Accuracy without splits:", acc_without_split)

# --- Part 3 - Task 2: Single Hidden Layer Perceptron with Sigmoid ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class SingleLayerPerceptron:
    def __init__(self, input_dim, hidden_neurons):
        self.W1 = np.random.randn(input_dim, hidden_neurons)
        self.b1 = np.zeros((1, hidden_neurons))
        self.W2 = np.random.randn(hidden_neurons, 1)
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, lr=0.1):
        m = len(y)
        dZ2 = self.A2 - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = dZ2 @ self.W2.T * sigmoid_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Toy dataset for binary classification
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])  # XOR problem

print("\nPerceptron Results (XOR problem):")
for h in [2, 4, 6]:
    model = SingleLayerPerceptron(input_dim=2, hidden_neurons=h)
    model.train(X, y, epochs=5000)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    print(f"Hidden neurons: {h}, Accuracy: {acc}")
