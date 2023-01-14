
from missile_env import MissileEnv
import warnings
import time
from rl_q_brain import QLearningTable

warnings.filterwarnings('ignore')

        
def update():
    for episode in range(10000):
        observation = env.reset()
        
        i = 0
        
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            
            observation_, reward, done, _ = env.step(action)
            
            observation = observation_
            
            i += 1
            
            if done:
                break
        print(f"Episode: {episode}")
        
        print("Game Over")
        
        
if __name__ == '__main__':
    env = MissileEnv()
    RL = QLearningTable(actions=list(range(env.NB_ACTIONS)))
    update()
