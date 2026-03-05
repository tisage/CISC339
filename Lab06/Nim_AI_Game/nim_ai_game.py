import gradio as gr
import random
import time

class Nim():
    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        self.piles[pile] -= count
        self.switch_player()

        if all(pile == 0 for pile in self.piles):
            self.winner = self.player

class NimAI():
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        return self.q.get((tuple(state), action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        new_q = old_q + self.alpha * ((reward + future_rewards) - old_q)
        self.q[(tuple(state), action)] = new_q

    def best_future_reward(self, state):
        max_reward = 0
        for sta, q in self.q.items():
            if sta[0] == tuple(state) and q > max_reward:
                max_reward = q
        return max_reward

    def choose_action(self, state, epsilon=True):
        max_reward = 0
        best_action = None
        available_moves = Nim.available_actions(state)

        for move in available_moves:
            q = self.q.get((tuple(state), move), 0)
            if q > max_reward:
                max_reward = q
                best_action = move

        if max_reward == 0:
            return random.choice(tuple(available_moves))

        if not epsilon:
            return best_action
        else:
            if random.random() < self.epsilon:
                return random.choice(tuple(available_moves))
            else:
                return best_action

def train(n):
    player = NimAI()
    for i in range(n):
        game = Nim()
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }
        while True:
            state = game.piles.copy()
            action = player.choose_action(game.piles)
            last[game.player]["state"] = state
            last[game.player]["action"] = action
            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )
    return player

# Pre-train AI
print("Training Reinforcement Learning AI (10,000 games)...")
trained_ai = train(10000)
print("Done training! Launching game server...")


def render_board(piles):
    """Renders the piles as emojis for visual appeal"""
    html = "<div style='font-size: 24px; font-family: monospace;'>"
    for i, p in enumerate(piles):
        html += f"<b>Pile {i}:</b> " + ("🔴" * p) + f" ({p})<br>"
    html += "</div>"
    return html

def start_game():
    game = Nim()
    human_player = 0 # Human goes first initially
    return game, render_board(game.piles), f"**Game Started!** You play first. Choose a pile and the number to take.", human_player

def make_move(game_state, pile, count, human_player):
    game = game_state
    
    if game.winner is not None:
        return game, render_board(game.piles), f"Game already over! Winner: {'Human' if game.winner == human_player else 'AI'}."
        
    # Validation
    try:
        pile = int(pile)
        count = int(count)
        if (pile, count) not in Nim.available_actions(game.piles):
            return game, render_board(game.piles), f"**INVALID MOVE!** You tried to take {count} from Pile {pile}. Only take from what is available in the pile!"
    except ValueError:
        return game, render_board(game.piles), "Please enter numbers for pile and count."

    # Human Move
    game.move((pile, count))
    
    if game.winner is not None:
        return game, render_board(game.piles), "🎉 **GAME OVER! YOU WIN!** (The AI took the last object)"

    # AI Turn Delay string for realism
    message = f"You took {count} from Pile {pile}. \n\n🤖 *AI is thinking...*\n"
    current_board = render_board(game.piles)
    
    # AI Move
    pile_ai, count_ai = trained_ai.choose_action(game.piles, epsilon=False)
    game.move((pile_ai, count_ai))
    
    if game.winner is not None:
         return game, render_board(game.piles), "☠️ **GAME OVER! AI WINS!** (You have to take the last object, sorry!)"
         
    message += f"**AI Move:** AI took {count_ai} from Pile {pile_ai}.\n\nYour Turn! Select a pile and count."
    return game, render_board(game.piles), message


# UI Components
with gr.Blocks(theme=gr.themes.Base()) as ui:
    gr.Markdown("# 🕹️ Reinforcement Learning: The Game of Nim")
    gr.Markdown("**Rules:** Start with 4 piles of objects. On your turn, take any number of objects from exactly one pile. **The player forced to take the last object loses!** Can you beat an AI trained on 10,000 matches?")
    
    game_state = gr.State()
    human_player = gr.State(value=0)
    
    with gr.Row():
        with gr.Column():
            board_display = gr.HTML(label="Game Board")
            status = gr.Markdown("Click 'Start New Game' to begin.")
            
        with gr.Column():
            gr.Markdown("### Your Move")
            pile_input = gr.Dropdown(choices=["0", "1", "2", "3"], label="Select Pile (0-3)")
            count_input = gr.Slider(minimum=1, maximum=7, step=1, label="Number of objects to take", value=1)
            submit_btn = gr.Button("Submit Move", variant="primary")
            restart_btn = gr.Button("Restart Game", variant="secondary")

    submit_btn.click(
        make_move, 
        inputs=[game_state, pile_input, count_input, human_player], 
        outputs=[game_state, board_display, status]
    )
    
    restart_btn.click(
        start_game,
        outputs=[game_state, board_display, status, human_player]
    )
    
    ui.load(start_game, outputs=[game_state, board_display, status, human_player])

if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1", inbrowser=True)
