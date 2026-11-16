# Day 7: thí nghiệm retrieval trên dev

## Cách setup và chạy 

Quay về thư mục gốc Day 7, tạo cấu hình local rồi sửa model path:

```powershell
cd ..\..\K5-L3B-DAY07-2A202604092-TranVanTuan-RetrievalContext
Copy-Item starter\.env.example starter\.env
```

Trong `starter\.env`, bảo đảm:

```dotenv
DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab
EMBEDDING_MODEL=C:\models\embeddinggemma-2
VECTOR_STRATEGY=A
```

`EMBEDDING_MODEL` phải trỏ tới đúng thư mục model đã cấu hình cho Day 6. `ANTHROPIC_API_KEY` chỉ cần điền key hợp lệ nếu chạy agent Claude; benchmark retrieval không gọi Claude. Không commit file `.env`.

Benchmark questions ở `starter\data\converted\questions.json` gồm dev/test và các câu thiếu evidence. File này được cho phép theo dõi Git riêng; giữ file khi commit để người khác có thể chạy benchmark sau khi clone. Chunk JSON không cần đưa vào repo: retriever đọc trực tiếp từ PostgreSQL.

### 5. Kiểm tra và chạy

Từ gốc Day 7:

```powershell
python -c "from starter.pgvector_retriever import PgVectorRetriever; print(len(PgVectorRetriever().search('Đất chợ dân sinh thuộc nhóm đất nào?')))"
python -m starter.benchmark_retrievers --questions starter\data\converted\questions.json --split dev --output starter\results\retrieval-dev\dev-k5-b1600 --top-k 5 --budget 1600
python -m starter.run_claude_agent "Đất chợ dân sinh thuộc nhóm đất nào?" --retrieval hybrid-rrf --max-searches 2 --top-k 5 --budget 1600
```

Lệnh benchmark không cần Claude key. Lệnh chạy agent cần PostgreSQL, model embedding tương thích và key Anthropic hợp lệ; API call có thể tính phí. Sau khi chốt trên dev, chạy test vào output riêng:

```powershell
python -m starter.benchmark_retrievers --questions starter\data\converted\questions.json --split test --output starter\results\retrieval-test\selected-k5-b1600 --top-k 5 --budget 1600
```

## Thiết lập và tiêu chí chọn

- Dữ liệu: 20 câu dev, gồm 18 câu có evidence và 2 câu thiếu evidence. Hai câu thiếu evidence không nằm trong mẫu số các metric retrieval.
- Corpus/index: cùng PostgreSQL + pgvector Day 6, bảng `rag_chunks_768`, strategy `A`, embedding `embeddinggemma-2`; không đổi dữ liệu, filter hay query.
- So sánh dense, TF-IDF lexical và hybrid RRF. RRF lấy candidate depth 20, `rrf_k=60`.
- Context được đóng gói theo thứ tự ranking, giới hạn theo số ký tự. `Hit@3` giữ cutoff cố định; `Hit@k` dùng top-k của cấu hình.
- Tiêu chí chốt trước test: ưu tiên Context Evidence Rate cao nhất, với Hit@3 không giảm so với baseline; cân nhắc mức tăng budget/context. Chỉ dev được dùng để chọn.
- Thời gian là median của `retriever.search` trên máy này, không gồm Claude/API. Chi phí Claude chưa đo; các benchmark này chỉ đánh giá retrieval và context.

## Kết quả

| Cấu hình | Phương pháp | Hit@1 | Evidence Hit@3 | Hit@k | Context Evidence Rate | Context TB (ký tự) | Median search (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline k3/b800 | Dense | 6/18 (33.3%) | 11/18 (61.1%) | 11/18 (61.1%) | 7/18 (38.9%) | 462 | 4011 |
| Baseline k3/b800 | TF-IDF | 8/18 (44.4%) | 10/18 (55.6%) | 10/18 (55.6%) | 8/18 (44.4%) | 417 | 23 |
| Baseline k3/b800 | Hybrid RRF | 7/18 (38.9%) | 12/18 (66.7%) | 12/18 (66.7%) | 7/18 (38.9%) | 422 | 4123 |
| Thử 1 k5/b800 | Dense | 6/18 (33.3%) | 11/18 (61.1%) | 12/18 (66.7%) | 7/18 (38.9%) | 477 | 4418 |
| Thử 1 k5/b800 | TF-IDF | 8/18 (44.4%) | 10/18 (55.6%) | 11/18 (61.1%) | 8/18 (44.4%) | 435 | 23 |
| Thử 1 k5/b800 | Hybrid RRF | 7/18 (38.9%) | 12/18 (66.7%) | 14/18 (77.8%) | 7/18 (38.9%) | 443 | 4241 |
| Thử 2 k5/b1200 | Dense | 6/18 (33.3%) | 11/18 (61.1%) | 12/18 (66.7%) | 11/18 (61.1%) | 762 | 3865 |
| Thử 2 k5/b1200 | TF-IDF | 8/18 (44.4%) | 10/18 (55.6%) | 11/18 (61.1%) | 9/18 (50.0%) | 801 | 22 |
| Thử 2 k5/b1200 | Hybrid RRF | 7/18 (38.9%) | 12/18 (66.7%) | 14/18 (77.8%) | 11/18 (61.1%) | 801 | 4013 |
| Thử 3 k5/b1600 | Dense | 6/18 (33.3%) | 11/18 (61.1%) | 12/18 (66.7%) | 11/18 (61.1%) | 1073 | 5812 |
| Thử 3 k5/b1600 | TF-IDF | 8/18 (44.4%) | 10/18 (55.6%) | 11/18 (61.1%) | 10/18 (55.6%) | 1104 | 25 |
| Thử 3 k5/b1600 | Hybrid RRF | 7/18 (38.9%) | 12/18 (66.7%) | 14/18 (77.8%) | 12/18 (66.7%) | 1117 | 5239 |

Hai cấu hình cải thiện chính:

1. **k5/b800** tăng Hybrid Hit@k từ 12/18 lên 14/18, nhưng Context Evidence Rate vẫn 7/18. Tăng top-k giúp tìm evidence ở hạng thấp hơn, nhưng chưa đảm bảo evidence được đưa vào context.
2. **k5/b1200** giữ Hybrid Hit@k 14/18 và tăng Context Evidence Rate lên 11/18, so với baseline 7/18.
3. **k5/b1600** đạt Context Evidence Rate cao nhất, 12/18; Hit@3 và Hit@5 không giảm so với hai thử nghiệm trước. Context trung bình tăng từ 801 lên 1117 ký tự so với k5/b1200.

## Hai lỗi dev đã đọc

- **D07 — bảng giá đất đã công bố:** baseline Hybrid không có gold trong top-3. Với k5, evidence `data:20:2` xuất hiện ở hạng 4 nhưng vẫn không vào context, kể cả budget 1600. Bộ đóng gói ưu tiên ba chunk trước theo ranking; các chunk đó chiếm phần lớn budget. Đây là lỗi chọn context sau retrieval.
- **D17 — thời hạn không sử dụng đất trồng cây lâu năm:** baseline Hybrid không có gold trong top-3. Với k5, chunk có evidence `data:77:0` lên hạng 4 nhưng bị loại khỏi context; context giữ `data:77:1` và các chunk xếp trước. Đây cũng là lỗi chọn context, không phải thiếu evidence trong corpus.

Các lỗi này cho thấy tăng top-k cải thiện ranking nhưng chưa giải quyết được việc chọn đoạn cho context. Cấu hình đã chọn vẫn có 6/18 câu answerable không có evidence trong context; cần xem tiếp trace/đánh giá câu trả lời thật.

## Cấu hình chốt trước khi chạy test

Chọn **Hybrid RRF, top-k=5, context budget=1600 ký tự** vì đạt Context Evidence Rate cao nhất trong các cấu hình đã thử (12/18), đồng thời Hit@3 giữ 12/18 và Hit@5 đạt 14/18. Budget cao hơn k5/b1200 thêm 400 ký tự, đổi lại context hit tăng một câu (11/18 lên 12/18). Lựa chọn được ghi nhận trước khi chạy test.

## Test sau khi chốt

| Cấu hình / phương pháp | Hit@1 | Evidence Hit@3 | Hit@k | Context Evidence Rate |
|---|---:|---:|---:|---:|
| Baseline k3/b800, Hybrid RRF | 2/6 (33.3%) | 3/6 (50.0%) | 3/6 (50.0%) | 2/6 (33.3%) |
| Đã chọn k5/b1600, Hybrid RRF | 2/6 (33.3%) | 3/6 (50.0%) | 3/6 (50.0%) | 3/6 (50.0%) |

Test có 8 câu (6 answerable, 2 unanswerable); câu unanswerable không tính vào mẫu số retrieval. Trên test nhỏ này, cấu hình đã chọn giữ Hit@3 và tăng context hit một câu; không dùng test để chỉnh cấu hình thêm.

Agent/Claude answer quality, citations, số token và API cost chưa được benchmark. Lần gọi Claude trước đó bị từ chối xác thực (401), nên kết quả ở đây chỉ là retrieval/context, không phải đánh giá câu trả lời của Claude.

## Lệnh tái tạo

Chạy từ thư mục gốc:

```powershell
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev/baseline-k3-b800 --top-k 3 --budget 800
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev/config-k5-b800 --top-k 5 --budget 800
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev/config-k5-b1200 --top-k 5 --budget 1200
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split dev --output starter/results/retrieval-dev/config-k5-b1600 --top-k 5 --budget 1600
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split test --output starter/results/retrieval-test/baseline-k3-b800 --top-k 3 --budget 800
python -m starter.benchmark_retrievers --questions starter/data/converted/questions.json --split test --output starter/results/retrieval-test/selected-hybrid-k5-b1600 --top-k 5 --budget 1600
```

Mỗi thư mục output có `summary.json` và `traces.json`. Xem lần lượt [baseline](../../starter/results/retrieval-dev/baseline-k3-b800/summary.json), [k5/b800](../../starter/results/retrieval-dev/config-k5-b800/summary.json), [k5/b1200](../../starter/results/retrieval-dev/config-k5-b1200/summary.json) và [k5/b1600](../../starter/results/retrieval-dev/config-k5-b1600/summary.json). Thư mục `starter/results/` đang bị Git ignore; báo cáo này ghi lại số liệu để nộp và đối chiếu.
