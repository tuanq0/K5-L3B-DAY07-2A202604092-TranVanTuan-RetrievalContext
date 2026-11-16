# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `claude_controller.py` | Gọi Claude API bằng tool calling, chuẩn hóa thành action search/answer |
| `run_claude_agent.py` | CLI chạy agent với Claude và một retrieval method |
| `pgvector_retriever.py` | Dense retriever kết nối PostgreSQL + pgvector theo schema Day 6 |
| `retrieval_methods.py` | TF-IDF lexical retriever và hybrid RRF trên cùng corpus PostgreSQL |
| `benchmark_retrievers.py` | So sánh dense, lexical TF-IDF và hybrid RRF trên một split |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Starter **chưa phải solution hoàn chỉnh**. Dense, lexical, hybrid retriever và Claude controller đã có; bạn vẫn cần cấu hình API key, xác nhận embedding khớp index, benchmark cấu hình và tự chấm câu trả lời/citation. Mock chỉ dùng để test code.

## 1. Kết nối agent

Trong `agentic.py`:

- `ClaudeController.decide`: adapter gọi Claude API và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `PgVectorRetriever.search`: adapter có sẵn cho PostgreSQL + pgvector theo schema Day 6; nếu schema hoặc embedding model khác, cập nhật cấu hình/adapter. Kết quả theo [định dạng dữ liệu](../data/README.md).

`run_claude_agent.py` nối controller và retriever để chạy; có thể thay thế bằng kiến trúc khác. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định, không phải giới hạn của đề. Xem `tests/test_agentic.py` và `tests/test_claude_controller.py`; các kết nối trong test là mock. Agent kiểm tra citation ID có trong context, nhưng bạn vẫn phải đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

### Chạy với Claude API

Đặt `ANTHROPIC_API_KEY` trong `starter/.env` (không commit file này). Mặc định
dùng `claude-haiku-5-5`; có thể đổi `CLAUDE_MODEL` và giới hạn đầu ra bằng
`CLAUDE_MAX_TOKENS`. Từ thư mục gốc chạy:

```powershell
python -m starter.run_claude_agent "Đất chợ dân sinh thuộc nhóm đất nào?"
```

Mặc định agent dùng hybrid RRF, tối đa 2 lượt tìm kiếm, top-3 và budget 800
ký tự. Chọn cách retrieval khác bằng `--retrieval dense` hoặc
`--retrieval lexical-tfidf`. Lệnh này gọi Claude API và có thể phát sinh phí.

## 2. Retriever PostgreSQL + pgvector

`pgvector_retriever.py` triển khai dense retrieval theo schema Day 6:
bảng `rag_chunks_768`, strategy `A` và filter metadata. Code đọc `DATABASE_URL`
và `EMBEDDING_MODEL` từ `starter/.env`. Đường dẫn trong `.env.example` minh họa
model local `embeddinggemma-2` của Day 6; nếu Day 6 dùng model khác, thay giá trị
đó. Embedding câu hỏi phải dùng đúng model và prompt đã tạo vector trong database. Đổi
`VECTOR_STRATEGY` nếu muốn tìm ở strategy khác đã được index.

Cài dependencies và chỉ copy file cấu hình mẫu nếu chưa có `.env`:

```powershell
python -m pip install -r starter/requirements.txt
if (-not (Test-Path starter/.env)) { Copy-Item starter/.env.example starter/.env }
```

Để PostgreSQL container Day 6 tiếp tục chạy và kiểm tra `DATABASE_URL` trong
`starter/.env`. Chạy thử dense retrieval từ thư mục gốc:

```powershell
python -c "from starter.pgvector_retriever import PgVectorRetriever; print(PgVectorRetriever().search('Đất chợ dân sinh thuộc nhóm đất nào?'))"
```

Dùng `PgVectorRetriever()` làm đối số `retriever` của `run_agent`. `Controller`
kết nối model AI vẫn cần được hiện thực riêng.

## 3. So sánh dense, lexical và hybrid

`PgTfidfRetriever` đọc cùng chunks/metadata từ bảng Day 6 và lập chỉ mục TF-IDF
trên máy. `HybridRRF` lấy tối đa 20 kết quả từ mỗi retriever rồi hợp nhất thứ
hạng bằng Reciprocal Rank Fusion với `rrf_k=60`; top-k cuối vẫn do tham số chạy
quy định. Gold chỉ được dùng sau retrieval để tính Hit@1, Hit@k và context hit; câu thiếu
evidence được loại khỏi mẫu số retrieval metrics.

Chạy benchmark trên dev:

```powershell
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev
```

Kết quả gồm `summary.json` và `traces.json`. Xem trace để so sánh chunk, score,
metadata và thời gian từng phương pháp. Chỉ chạy `--split test` sau khi đã chốt
cấu hình trên dev, và ghi ra thư mục output riêng. Mỗi cấu hình báo Hit@1,
Evidence Hit@3 (cutoff cố định để so sánh) và Hit@k (k cấu hình); context hit
được tính riêng theo context budget đã chọn.

## 4. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 5. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 6. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
