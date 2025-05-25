# simple-djssp-llm
## Cấu trúc mã nguồn
* [model.py](./model.py): Chứa các mô hình đại diện cho các HDR  hiện đang dùng CodeSegmentHDR.
* [problem.py](./problem.py): Chứa các mô hình về bài toán lập lịch động, bao gồm Operation, Job, Machine và các Terminal được sử dụng.
* [llm_support.py](./llm_support.py): Chứa các lớp và phương thức để gọi và lấy response từ LLM (đang dùng LLM trên OpenRouter, có các bản free)
* [evaluate.py](./evaluate.py): Chứa Simulator để chạy mô phỏng với 1 HDR và 1 Problem và các phương thức đánh giá fitness của 1 HDR (simulaton-based)
* [basic_evo.py](./basic_evo.py): Chứa các lớp cơ bản của tính toán tiến hóa như Individual, Population và định
nghĩa interface Operator (các toán tử tiến hóa được sử dụng)
* [llm_evo.py](./llm_evo.py): Triển khai các toán tử Se-Evo, phần chính, chứa các LLM-Base Operator và các Reflection Operator.
* [prompt_template.py](./prompt_template.py): Chứa các string template của các prompt được sử dụng trong các LLM-Base Operator
* [template.txt](./template.txt): Chứa template của 1 code segment nên được trả về từ LLM.
## Môi trường phát triển
- Ngôn ngữ: Python (3.13.3)
- IDE: VS Code
- OS: Windows
- Hỗ trợ chạy thử nghiệm: Kaggle Notebook
## Chạy thử nghiệm
Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```
Chạy file `main.py`.
```bash
python -u main.py
```
Hoặc tạo 1 file tương tự với các bước:
1. Tạo logger (tùy chọn)
```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Định dạng chung
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler ghi vào file
file_handler = logging.FileHandler(f'process_{datetime.now().strftime("%Y_%m_%d")}.log')
file_handler.setFormatter(formatter)

# Handler ghi ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Thêm cả 2 handler vào logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

2. Tạo Problem.
Hiện đã cung cấp cả 2 phương thức tạo problem: Random hoàn toàn (`problem.random_generate`) và Random một phần (`problem.custom_generate`). Ở Random 1 phần có thể chọn phân phối thời gian đến (uniform, burst, v.v) cũng như phân phối thời gian xử lý, cùng với deadline factor.
Sau đây là ví dụ tạo Problem.
```python
# Nên đặt random.seed cho giống nhau
random.seed(42)
problem = Problem(AVAIABLE_TERMINALS, pool_size=5)
problem.custom_generate(num_jobs=15, max_oprs_each_job=3,
                        num_machines=4, max_arr_time=20,
                        arrival_type='uniform', proc_dist='uniform', deadline_factor=1.4)
```

3. Thêm API Key và tạo LLM.
Thêm API Key cần thiết vào file `config/llm_core.json` và chỉnh đường dẫn. Hiện hỗ trợ OpenRouterAPI và GoogleAIStudioAPI.
- Với OpenRouterAPI, thêm các trường sau:
```json
{
    "OPEN_ROUTER_PROVISION_KEY": "Your-provison-key",
    "OPEN_ROUTER_API_KEY": "your-api-key",
    "OPEN_ROUTER_API_KEY_HASH": "your-api-key-hash"
}
```
 trong đó trường `OPEN_ROUTER_PROVISION_KEY` là bắt buộc. 2 trường khác sẽ được tự khởi tạo khi bạn dùng lần đầu và sẽ được dùng lại cho các lần kế tiếp, hoặc có thể tự set nếu đã có API key.
- Với Google AI Studio API, thêm trường
```json
{
    "GOOGLE_AI_API_KEY": "your-api-key"
}
```

Sau đó, tùy vào chọn API nào mà gọi class tương ứng.
```python
# Nếu dùng Open Router API
llm_model = OpenRouterLLM('deepseek', 'deepseek-r1-zero', free=True, timeout=(60, 600),
                          core_config='config/llm_core.json',
                          runtime_config='config/llm_runtime.json')

# Nếu dùng Google AI Studio
llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='config/llm_core.json',
                              runtime_config='config/llm_runtime.json')
```

4. Tạo các Operator (có sẵn hoặc tự tạo thêm, miễn tuân thủ prototype)
```python
# Create Operator
init_opr = LLMInitOperator(problem, llm_model, pt.INIT_IND_PROMPT_TEMPLATE)
crossover_opr = LLMCrossoverOperator(problem, llm_model, pt.CROSSOVER_PROMPT_TEMPLATE)
mutation_opr = LLMMutationOperator(problem, llm_model, pt.MUTATION_PROMPT_TEMPLATE)
replace_opr = TopKElitismReplaceOperator(problem, k=2)
```

5. Khởi tạo bộ mô phỏng, gắn các HDR mặc định
```python
# Nếu dùng Simulation
hdr1 = CodeSegmentHDR(code="""
def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    return -jnpt                   
""")

hdr2 = CodeSegmentHDR(code="""
def hdr(jnpt=0.0, japt=0.0, jrt=0.0, jro=0.0, jwt=0.0, jat=0.0, jd=0.0, jcd=0.0, js=0.0, jw=0.0, ml=0.0, mr=0.0, mrel=0.0, mpr=0.0, mutil=0.0, tnow=0.0, util=0.0, avgwt=0.0):
    return mrel
""")

evaluator = SimulationBaseEvaluator(Simulator(problem, hdr1, hdr2))
```

6. Tạo Engine solve
```python 
engine = LLMEvoEngine(problem, init_opr, crossover_opr, mutation_opr, replace_opr,
                      evaluator, max_retries=3)
)
```
7. Run Engine.
```python
best = engine.solve(max_gen=10, init_size=12, template_dir='template.txt')

if best is None:
    print("Not found sol!")
else:
    print("Best: ")
    print(str(best.chromosome))
    print(best.fitness)
    print(f"Time: {engine.solve_time:.2f}s")
    os.makedirs('best_solution', exist_ok=True)
    best.chromosome.save(f'best_after_gen_{engine.gen}.py')
```
8. Đóng kết nối LLM (nên có)
```python
llm_model.close()
```