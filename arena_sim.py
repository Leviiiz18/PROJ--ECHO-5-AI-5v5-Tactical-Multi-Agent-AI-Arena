import os
import math
import random
import sys
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- Dynamic Constants (Will be set at runtime) ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
ARENA_MARGIN = 20
MAX_STEPS = 3000   # Increased for longer, more strategic rounds

# Tactical Color Palette
COLOR_BG = (12, 14, 12)
COLOR_WALL = (35, 40, 35)
COLOR_TEAM_A = (65, 150, 255)
COLOR_TEAM_B = (255, 110, 40)
COLOR_BULLET = (255, 255, 180)
COLOR_HEALTH_BAR = (60, 200, 60)
COLOR_DAMAGED = (200, 50, 50)
COLOR_TEXT = (210, 215, 190)
COLOR_MUZZLE = (255, 190, 40)
COLOR_GRID = (25, 28, 25)

# Physics & Logic
AGENT_RADIUS = 15
MOVE_SPEED = 3.6
ROTATION_SPEED = 0.15 
BULLET_SPEED = 18.0
MAX_HEALTH = 100.0
FIRE_COOLDOWN = 10
BULLET_DAMAGE = 25.0

# RL Rewards (High-Fidelity Tuning)
REWARD_KILL = 50.0
REWARD_HIT = 15.0            
REWARD_OBSTACLE_HIT = -2.0   
REWARD_MISS = -2.0           
REWARD_DEATH = -10.0
REWARD_WIN = 300.0           
REWARD_LOSS = -100.0         
REWARD_TIME_PENALTY = 0.0    
REWARD_FACING_ENEMY = 2.0    
REWARD_SPIN_PENALTY = -0.5   

class Particle:
    def __init__(self, x, y, dx, dy, life, color, size=3):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self):
        self.x += self.dx; self.y += self.dy
        self.life -= 1
        return self.life > 0

    def draw(self, surface, offset=(0,0)):
        alpha = int((self.life / self.max_life) * 255)
        p_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (*self.color, alpha), (self.size, self.size), self.size)
        surface.blit(p_surf, (int(self.x - self.size + offset[0]), int(self.y - self.size + offset[1])), special_flags=pygame.BLEND_RGB_ADD)

class Bullet:
    def __init__(self, x, y, dx, dy, owner_id, team):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.owner_id = owner_id
        self.team = team
        self.active = True
        self.trail = []
        self.hit_something = False

    def update(self):
        self.trail.append((self.x, self.y))
        if len(self.trail) > 6: self.trail.pop(0)
        self.x += self.dx; self.y += self.dy
        if not (0 <= self.x <= SCREEN_WIDTH and 0 <= self.y <= SCREEN_HEIGHT): self.active = False

    def draw(self, surface, offset=(0,0)):
        if not self.active: return
        for i, pos in enumerate(self.trail):
            alpha = int((i / len(self.trail)) * 150)
            sz = 2 + (i/len(self.trail))*2
            p_surf = pygame.Surface((sz*2, sz*2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, (*COLOR_BULLET, alpha), (sz, sz), sz)
            surface.blit(p_surf, (int(pos[0]-sz+offset[0]), int(pos[1]-sz+offset[1])), special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.circle(surface, COLOR_BULLET, (int(self.x + offset[0]), int(self.y + offset[1])), 3)

class Agent:
    def __init__(self, id, team, x, y, color):
        self.id, self.team = id, team
        self.x, self.y = x, y
        self.color = color
        self.health = MAX_HEALTH
        self.visual_health = MAX_HEALTH
        self.angle = 0
        self.last_fire_time = 0
        self.is_alive = True
        self.radius = AGENT_RADIUS

    def reset(self, x, y):
        self.x, self.y = x, y
        self.health = MAX_HEALTH
        self.visual_health = MAX_HEALTH
        self.is_alive = True
        self.angle = random.uniform(0, math.pi*2)
        self.last_fire_time = 0

    def move(self, move_dir, rot_dir, obstacles):
        if not self.is_alive: return
        self.angle += rot_dir * ROTATION_SPEED
        dx, dy = math.cos(self.angle)*move_dir*MOVE_SPEED, math.sin(self.angle)*move_dir*MOVE_SPEED
        new_x = max(ARENA_MARGIN+self.radius, min(SCREEN_WIDTH-ARENA_MARGIN-self.radius, self.x + dx))
        new_y = max(ARENA_MARGIN+self.radius, min(SCREEN_HEIGHT-ARENA_MARGIN-self.radius, self.y + dy))
        rect = pygame.Rect(new_x - self.radius, new_y - self.radius, self.radius*2, self.radius*2)
        if not any(rect.colliderect(obs) for obs in obstacles):
            self.x, self.y = new_x, new_y

    def draw(self, surface, offset=(0,0)):
        if not self.is_alive: return
        glow_surf = pygame.Surface((self.radius*4, self.radius*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color, 40), (self.radius*2, self.radius*2), self.radius*1.5)
        surface.blit(glow_surf, (int(self.x-self.radius*2+offset[0]), int(self.y-self.radius*2+offset[1])), special_flags=pygame.BLEND_RGB_ADD)
        pygame.draw.circle(surface, self.color, (int(self.x+offset[0]), int(self.y+offset[1])), self.radius)
        lx, ly = self.x+math.cos(self.angle)*20, self.y+math.sin(self.angle)*20
        pygame.draw.line(surface, (255,255,255), (self.x+offset[0], self.y+offset[1]), (lx+offset[0], ly+offset[1]), 4)
        if self.visual_health > self.health: self.visual_health -= 0.5
        bx, by = self.x-15+offset[0], self.y-25+offset[1]
        pygame.draw.rect(surface, (20,20,20), (bx, by, 30, 5))
        pygame.draw.rect(surface, COLOR_DAMAGED, (bx, by, int(30*(self.visual_health/MAX_HEALTH)), 5))
        pygame.draw.rect(surface, COLOR_HEALTH_BAR, (bx, by, int(30*(self.health/MAX_HEALTH)), 5))

class ActorCritic(nn.Module):
    def __init__(self, state_dim, move_dim, rot_dim, shoot_dim):
        super(ActorCritic, self).__init__()
        self.base = nn.Sequential(nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU())
        self.move_h = nn.Linear(128, move_dim); self.rot_h = nn.Linear(128, rot_dim)
        self.shoot_h = nn.Linear(128, shoot_dim); self.val_h = nn.Linear(128, 1)
    def forward(self, state):
        x = self.base(state)
        return torch.softmax(self.move_h(x), -1), torch.softmax(self.rot_h(x), -1), torch.softmax(self.shoot_h(x), -1), self.val_h(x)

class RLAgent:
    def __init__(self, id, lr=1.5e-4):
        self.id, self.state_dim = id, 19
        self.policy = ActorCritic(self.state_dim, 3, 3, 2)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(self.state_dim, 3, 3, 2); self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = {'states':[], 'actions':[], 'logprobs':[], 'rewards':[], 'is_terminals':[]}
        self.fitness = 0.0
        # Round Stats
        self.kills = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.elite = False
        self.mutated = False

    def select_action(self, state, eval_mode=False):
        state_t = torch.from_numpy(state).float()
        with torch.no_grad():
            # In deep RL, policy_old is the stableactor. We use it for training.
            # In evaluation, we use the primary 'policy' which is most up-to-date.
            m_p, r_p, s_p, _ = (self.policy if eval_mode else self.policy_old)(state_t)
        
        # We use Categorical sampling for both to keep the 'crisp' organic movement.
        # This prevents the 'stuck in the corner' behavior seen in purely deterministic argmax.
        m_d, r_d, s_d = Categorical(m_p), Categorical(r_p), Categorical(s_p)
        m_a, r_a, s_a = m_d.sample(), r_d.sample(), s_d.sample()
        
        if not eval_mode:
            self.memory['states'].append(state_t)
            self.memory['actions'].append(torch.tensor([m_a, r_a, s_a]))
            self.memory['logprobs'].append(m_d.log_prob(m_a)+r_d.log_prob(r_a)+s_d.log_prob(s_a))
        
        return [m_a.item(), r_a.item(), s_a.item()]

    def update(self):
        if not self.memory['rewards']: return
        rewards = []; discount = 0
        for r, term in zip(reversed(self.memory['rewards']), reversed(self.memory['is_terminals'])):
            if term: discount = 0
            discount = r + (0.99 * discount); rewards.insert(0, discount)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        st, act, lp = torch.stack(self.memory['states']).detach(), torch.stack(self.memory['actions']).detach(), torch.stack(self.memory['logprobs']).detach()
        for _ in range(4):
            m_p, r_p, s_p, vals = self.policy(st)
            m_d, r_d, s_d = Categorical(m_p), Categorical(r_p), Categorical(s_p)
            clp = m_d.log_prob(act[:,0]) + r_d.log_prob(act[:,1]) + s_d.log_prob(act[:,2])
            adv = rewards - vals.detach().squeeze()
            ratio = torch.exp(clp - lp)
            loss = -torch.min(ratio*adv, torch.clamp(ratio,0.8,1.2)*adv) + 0.5*nn.MSELoss()(vals.squeeze(), rewards) - 0.01*(m_d.entropy()+r_d.entropy()+s_d.entropy())
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict()); self.memory = {k:[] for k in self.memory}

    def mutate(self, source, rate=0.04):
        self.policy.load_state_dict(source.policy.state_dict())
        with torch.no_grad():
            for p in self.policy.parameters():
                if len(p.shape)>0: p.add_(torch.randn_like(p)*rate)
        # CRITICAL: Always sync policy_old after mutation
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1.5e-4)
        self.mutated = True

class TacticsArena:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        global SCREEN_WIDTH, SCREEN_HEIGHT
        SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.display.set_caption("Tactical Multi-Agent Arena (GA + RL)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Verdana", 14); self.font_big = pygame.font.SysFont("Verdana", 24, bold=True)
        self.font_title = pygame.font.SysFont("Verdana", 36, bold=True)
        
        self.state = "BATTLE" # BATTLE, SUMMARY
        self.summary_timer = 0
        self.summary_data = []
        
        self.obstacles = self.generate_random_obstacles(10)
        self.agents = [Agent(i, "A" if i < 5 else "B", 0,0, COLOR_TEAM_A if i < 5 else COLOR_TEAM_B) for i in range(10)]
        self.brains = [RLAgent(i) for i in range(10)]
        self.bullets, self.particles = [], []
        self.training_mode = True
        self.paused = False
        self.message = ""; self.message_timer = 0
        self.episode, self.steps, self.shake, self.generation = 0, 0, 0, 1
        self.wins = {"A":0, "B":0}
        self.vignette = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for i in range(150): pygame.draw.rect(self.vignette, (0,0,0, int(255*(i/150)**2)), (i,i,SCREEN_WIDTH-i*2, SCREEN_HEIGHT-i*2), 1)
        self.load_models()

    def generate_random_obstacles(self, count):
        obs = []
        sx_min, sx_max = 300, SCREEN_WIDTH - 300
        sy_min, sy_max = 100, SCREEN_HEIGHT - 100
        for _ in range(count):
            w, h = random.randint(60, 130), random.randint(60, 130)
            obs.append(pygame.Rect(random.randint(sx_min, sx_max-w), random.randint(sy_min, sy_max-h), w, h))
        return obs

    def reset(self):
        self.bullets, self.particles = [], []
        self.steps = 0
        for a in self.agents:
            x = random.randint(50, 250) if a.team == "A" else random.randint(SCREEN_WIDTH-250, SCREEN_WIDTH-50)
            a.reset(x, random.randint(50, SCREEN_HEIGHT-50))
        for b in self.brains: 
            b.kills = 0; b.shots_fired = 0; b.shots_hit = 0
            b.elite = False; b.mutated = False
        self.episode += 1

    def evolve(self):
        self.generation += 1
        for team in ["A", "B"]:
            t_b = [self.brains[i] for i in range(10) if self.agents[i].team == team]
            t_b.sort(key=lambda b: b.fitness, reverse=True)
            for i, b in enumerate(t_b):
                if i < 2: 
                    b.elite = True
                else:
                    # Everyone else is mutated from one of the two elites
                    b.mutate(random.choice(t_b[:2]))
            for b in t_b: b.fitness = 0.0
        self.save_models()

    def get_obs(self, agent):
        if not agent.is_alive: return np.zeros(19, dtype=np.float32)
        obs = [self.cast_ray(agent.x, agent.y, agent.angle + (i*math.pi/4))/400.0 for i in range(8)]
        enemies = [o for o in self.agents if o.team != agent.team and o.is_alive]
        def rd(o): return [(o.x-agent.x)/SCREEN_WIDTH, (o.y-agent.y)/SCREEN_HEIGHT, math.hypot(o.x-agent.x, o.y-agent.y)/1000.0] if o else [0,0,0]
        ne = min(enemies, key=lambda o: math.hypot(o.x-agent.x, o.y-agent.y), default=None)
        obs.extend(rd(ne) + [0,0,0]) # Simplified for speed
        los = 0.0
        if ne:
            ang, dist = math.atan2(ne.y-agent.y, ne.x-agent.x), math.hypot(ne.x-agent.x, ne.y-agent.y)
            if self.cast_ray(agent.x, agent.y, ang) >= dist-5: los = 1.0
        obs.extend([math.cos(agent.angle), math.sin(agent.angle), agent.health/MAX_HEALTH, 1.0 if agent.last_fire_time <= 0 else 0, los])
        return np.array(obs, dtype=np.float32)

    def cast_ray(self, x, y, angle):
        dx, dy = math.cos(angle), math.sin(angle)
        for d in range(10, 400, 15):
            px, py = x + dx*d, y + dy*d
            if not(0<=px<=SCREEN_WIDTH and 0<=py<=SCREEN_HEIGHT) or any(obs.collidepoint(px, py) for obs in self.obstacles): return d
        return 400

    def set_message(self, text, duration=120): self.message, self.message_timer = text, duration
    def save_models(self):
        if not os.path.exists("./models"): os.makedirs("./models")
        for i, b in enumerate(self.brains): torch.save(b.policy.state_dict(), f"./models/agent_{i}.pth")
    def load_models(self):
        if os.path.exists("./models/agent_0.pth"):
            try:
                for i, b in enumerate(self.brains): 
                    b.policy.load_state_dict(torch.load(f"./models/agent_{i}.pth"))
                    # CRITICAL: Sync policy_old after loading weights
                    b.policy_old.load_state_dict(b.policy.state_dict())
                self.set_message("MODELS LOADED & SYNCED")
            except: 
                self.set_message("MODEL MISMATCH - STARTING FRESH")

    def spawn_sparks(self, x, y, color, n=8):
        for _ in range(n): self.particles.append(Particle(x, y, random.uniform(-4,4), random.uniform(-4,4), random.randint(15,30), color, 2))

    def collect_summary_data(self):
        self.summary_data = []
        for i in range(10):
            b = self.brains[i]
            acc = (b.shots_hit / b.shots_fired * 100) if b.shots_fired > 0 else 0
            self.summary_data.append({
                'id': i, 'team': self.agents[i].team, 'kills': b.kills, 'acc': acc, 
                'fitness': int(b.fitness), 'elite': b.elite, 'mutated': b.mutated
            })

    def run(self):
        self.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.save_models(); pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if self.state == "SUMMARY": self.state = "BATTLE"; self.reset(); continue
                    if event.key == pygame.K_t: self.training_mode = not self.training_mode; self.set_message("TRAINING TOGGLED")
                    if event.key == pygame.K_p: self.paused = not self.paused
                    if event.key == pygame.K_s: self.save_models(); self.set_message("MODELS SAVED")
                    if event.key == pygame.K_r: self.reset(); self.generation=1; self.wins={"A":0,"B":0}

            if self.state == "SUMMARY":
                self.draw_summary_screen()
                self.summary_timer -= 1
                if self.summary_timer <= 0:
                    self.state = "BATTLE"
                    self.reset()
                self.clock.tick(60); continue

            if self.paused: self.draw_frame([]); self.clock.tick(60); continue

            ids = [a.id for a in self.agents if a.is_alive]
            obs = {i: self.get_obs(self.agents[i]) for i in ids}
            acts = {i: self.brains[i].select_action(obs[i], not self.training_mode) for i in ids}
            rwds = {i: 0.0 for i in ids}

            for i in ids:
                a, (mv, rot, sh) = self.agents[i], acts[i]
                ne = min([o for o in self.agents if o.team != a.team and o.is_alive], key=lambda o: math.hypot(o.x-a.x, o.y-a.y), default=None)
                if ne and abs((math.atan2(ne.y-a.y, ne.x-a.x)-a.angle + math.pi)%(2*math.pi)-math.pi)<0.4: rwds[i] += REWARD_FACING_ENEMY
                if rot != 1: rwds[i] += REWARD_SPIN_PENALTY
                a.move(mv-1, rot-1, self.obstacles)
                if sh == 1 and a.last_fire_time <= 0 and ne:
                    d = math.hypot(ne.x-a.x, ne.y-a.y)
                    self.bullets.append(Bullet(a.x, a.y, ((ne.x-a.x)/d)*BULLET_SPEED, ((ne.y-a.y)/d)*BULLET_SPEED, i, a.team))
                    a.last_fire_time = FIRE_COOLDOWN; self.brains[i].shots_fired += 1
                    self.spawn_sparks(a.x+math.cos(a.angle)*15, a.y+math.sin(a.angle)*15, COLOR_MUZZLE, 5)
                if a.last_fire_time > 0: a.last_fire_time -= 1
                rwds[i] += REWARD_TIME_PENALTY

            for b in self.bullets[:]:
                b.update()
                if not b.active:
                    if b.owner_id in rwds and not b.hit_something: rwds[b.owner_id] += REWARD_MISS
                    self.bullets.remove(b); continue
                if any(o.collidepoint(b.x, b.y) for o in self.obstacles):
                    if b.owner_id in rwds: rwds[b.owner_id] += REWARD_OBSTACLE_HIT
                    self.spawn_sparks(b.x, b.y, COLOR_WALL, 6); self.bullets.remove(b); continue
                for t in self.agents:
                    if t.is_alive and t.team != b.team and math.hypot(b.x-t.x, b.y-t.y) < t.radius:
                        t.health -= BULLET_DAMAGE; b.hit_something = True
                        if b.owner_id in rwds: rwds[b.owner_id] += REWARD_HIT; self.brains[b.owner_id].shots_hit += 1
                        self.spawn_sparks(b.x, b.y, t.color, 10); self.shake = 5
                        if t.health <= 0:
                            t.is_alive = False
                            if b.owner_id in rwds: rwds[b.owner_id] += REWARD_KILL; self.brains[b.owner_id].kills += 1
                            if t.id in rwds: rwds[t.id] += REWARD_DEATH
                            self.shake = 10
                        b.active = False; break
                if not b.active: self.bullets.remove(b)

            if self.training_mode:
                for i in ids: self.brains[i].memory['rewards'].append(rwds[i]); self.brains[i].memory['is_terminals'].append(False); self.brains[i].fitness += rwds[i]

            aa, ab = any(a.is_alive for a in self.agents if a.team=="A"), any(a.is_alive for a in self.agents if a.team=="B")
            if not aa or not ab or self.steps > MAX_STEPS:
                w = "A" if aa else "B"
                self.wins[w] += 1
                if self.training_mode:
                    for a in self.agents:
                        if self.brains[a.id].memory['rewards']:
                            r = (REWARD_WIN if a.team==w else REWARD_LOSS)
                            self.brains[a.id].memory['rewards'][-1]+=r; self.brains[a.id].memory['is_terminals'][-1]=True; self.brains[a.id].fitness+=r
                    self.evolve()
                self.collect_summary_data()
                self.state = "SUMMARY"; self.summary_timer = 540 # 9 Seconds

            self.steps += 1
            if self.training_mode and self.steps % 2000 == 0:
                for br in self.brains: br.update()
            self.draw_frame(ids)
            self.clock.tick(60)

    def draw_summary_screen(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 15, 10, 220))
        self.screen.blit(overlay, (0,0))
        
        y_off = 100
        title = self.font_title.render("AFTER ACTION REPORT", True, (0, 255, 150))
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, y_off))
        y_off += 60
        
        headers = ["AGENT", "TEAM", "KILLS", "ACCURACY", "RANK/STATUS"]
        h_x = [SCREEN_WIDTH//2 - 300, SCREEN_WIDTH//2 - 180, SCREEN_WIDTH//2 - 50, SCREEN_WIDTH//2 + 80, SCREEN_WIDTH//2 + 220]
        for i, h in enumerate(headers):
            self.screen.blit(self.font_big.render(h, True, COLOR_TEXT), (h_x[i], y_off))
        
        y_off += 40
        for d in self.summary_data:
            color = COLOR_TEAM_A if d['team'] == "A" else COLOR_TEAM_B
            text_color = COLOR_TEXT
            status = "MUTATED (Evolved)" 
            if d['elite']: 
                status = "ELITE (Winner)"
                text_color = (255, 215, 0)
            else:
                text_color = (255, 120, 40)
            
            row = [f"Agent {d['id']}", d['team'], f"{d['kills']}", f"{d['acc']:.1f}%", status]
            for i, val in enumerate(row):
                self.screen.blit(self.font.render(val, True, text_color if i == 4 else (color if i < 2 else COLOR_TEXT)), (h_x[i], y_off))
            y_off += 25
        
        y_off += 40
        footer = self.font.render(f"NEXT GENERATION STARTING IN {self.summary_timer//60 + 1}s...", True, (255,255,100))
        self.screen.blit(footer, (SCREEN_WIDTH//2 - footer.get_width()//2, y_off))
        pygame.display.flip()

    def draw_frame(self, ids):
        off = (random.randint(-self.shake, self.shake), random.randint(-self.shake, self.shake)) if self.shake > 0 else (0,0)
        self.shake = max(0, self.shake-1)
        self.screen.fill(COLOR_BG)
        for x in range(0, SCREEN_WIDTH, 40): pygame.draw.line(self.screen, COLOR_GRID, (x+off[0],0), (x+off[0], SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, 40): pygame.draw.line(self.screen, COLOR_GRID, (0,y+off[1]), (SCREEN_WIDTH, y+off[1]))
        for o in self.obstacles: pygame.draw.rect(self.screen, COLOR_WALL, o.move(off)); pygame.draw.rect(self.screen, (70,80,70), o.move(off), 2)
        self.particles = [p for p in self.particles if p.update()]; [p.draw(self.screen, off) for p in self.particles]
        for b in self.bullets: b.draw(self.screen, off)
        for a in self.agents: a.draw(self.screen, off)
        self.screen.blit(self.vignette, (0,0), special_flags=pygame.BLEND_RGBA_SUB)
        self.screen.blit(self.font_big.render(f"SCORE A: {self.wins['A']} | B: {self.wins['B']}", True, COLOR_TEXT), (20, 20))
        
        timer_w = int((SCREEN_WIDTH - 40) * (self.steps / MAX_STEPS))
        pygame.draw.rect(self.screen, (30,30,30), (20, SCREEN_HEIGHT-30, SCREEN_WIDTH-40, 10))
        pygame.draw.rect(self.screen, (150,50,50), (20, SCREEN_HEIGHT-30, timer_w, 10))
        
        m = [f"GENERATION: {self.generation}", f"UNITS ACTIVE: {len(ids)}", f"MODE: {'TRAINING' if self.training_mode else 'EVALUATION'}", "[T]oggle [P]ause [S]AVE [R]ESET"]
        for i, text in enumerate(m): self.screen.blit(self.font.render(text, True, COLOR_TEXT), (20, 60 + i*22))
        if self.message_timer > 0:
            surf = self.font.render(self.message, True, (255,255,100))
            self.screen.blit(surf, (SCREEN_WIDTH//2-surf.get_width()//2, 50)); self.message_timer -= 1
        pygame.display.flip()

if __name__ == "__main__":
    TacticsArena().run()
