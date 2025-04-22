# ...existing code...

class Cube:
    def __init__(self, position, size=1):
        self.position = np.array(position, dtype=float)
        self.size = size
        self.velocity = np.random.randn(2) * 2  # 随机初速度
        self.color = (255, 255, 255)  # 默认白色
        
    def update(self, dt):
        # 更新位置
        self.position += self.velocity * dt
        
        # 碰到边界反弹
        for i in range(2):
            if self.position[i] <= 0 or self.position[i] >= 100 - self.size:
                self.velocity[i] *= -1
                self.position[i] = np.clip(self.position[i], 0, 100 - self.size)
                
    def get_aabb(self):
        return AABB(self.position, self.position + self.size)

class Visualization:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("八叉树碰撞检测可视化")
        self.clock = pygame.time.Clock()
        
        # 创建2000个立方体
        self.cubes = [Cube(np.random.uniform(0, 99, 2)) for _ in range(2000)]
        self.scale = 8  # 显示缩放因子
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.rebuild_time = 0
        self.check_time = 0
        self.intersects_check_count = 0
        
    def update(self, dt):
        # 更新所有立方体位置
        for cube in self.cubes:
            cube.update(dt)
        
        # 重置碰撞状态
        for cube in self.cubes:
            cube.color = (255, 255, 255)
        
        # 构建八叉树
        rebuild_start = time.time()
        bounds = AABB(np.array([0, 0]), np.array([100, 100]))
        octree = Octree(bounds)
        
        for i, cube in enumerate(self.cubes):
            octree.insert((i, cube), cube.get_aabb())
        
        self.rebuild_time = time.time() - rebuild_start
        
        # 碰撞检测
        check_start = time.time()
        self.intersects_check_count = 0
        collisions = set()
        
        def check_node_collisions(node):
            # 检查当前节点中物体之间的碰撞
            for i, (idx1, cube1) in enumerate(node.objects):
                for j, (idx2, cube2) in enumerate(node.objects[i+1:], i+1):
                    self.intersects_check_count += 1
                    if cube1.get_aabb().intersects(cube2.get_aabb()):
                        collisions.add(tuple(sorted((idx1, idx2))))
            
            # 检查子节点
            if node.children:
                for child in node.children:
                    check_node_collisions(child)
        
        check_node_collisions(octree.root)
        self.check_time = time.time() - check_start
        
        # 标记碰撞的立方体
        for idx1, idx2 in collisions:
            self.cubes[idx1].color = (255, 0, 0)
            self.cubes[idx2].color = (255, 0, 0)
    
    def draw(self):
        self.screen.fill((0, 0, 0))  # 黑色背景
        
        # 绘制所有立方体
        for cube in self.cubes:
            pos = cube.position * self.scale
            size = cube.size * self.scale
            pygame.draw.rect(self.screen, cube.color, 
                           (pos[0], pos[1], size, size))
        
        # 显示性能指标
        self.frame_count += 1
        if self.frame_count % 60 == 0:  # 每秒更新一次
            current_time = time.time()
            fps = self.frame_count / (current_time - self.start_time)
            
            font = pygame.font.Font(None, 36)
            stats = [
                f"FPS: {fps:.1f}",
                f"重建时间: {self.rebuild_time*1000:.1f}ms",
                f"检测时间: {self.check_time*1000:.1f}ms",
                f"碰撞检查次数: {self.intersects_check_count}"
            ]
            
            for i, text in enumerate(stats):
                surface = font.render(text, True, (255, 255, 255))
                self.screen.blit(surface, (10, 10 + i*30))
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            dt = self.clock.tick(60) / 1000.0  # 转换为秒
            self.update(dt)
            self.draw()
        
        pygame.quit()

if __name__ == "__main__":
    vis = Visualization()
    vis.run()