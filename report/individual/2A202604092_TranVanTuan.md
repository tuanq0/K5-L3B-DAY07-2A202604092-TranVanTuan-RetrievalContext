# Day 7: đánh giá Claude agent trên test

## Cách setup và chạy từ lúc git clone

Ví dụ dùng Windows PowerShell. Cần Git, Python 3.11+, quyền truy cập PostgreSQL đã có index cần cho Day 7, thư mục embedding model local 768 chiều và Anthropic API key.

```powershell
cd $HOME
git clone https://github.com/tuanq0/K5-L3B-DAY07-2A202604092-TranVanTuan-RetrievalContext.git
cd $HOME\K5-L3B-DAY07-2A202604092-TranVanTuan-RetrievalContext
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r starter\requirements.txt
Copy-Item starter\.env.example starter\.env
```

Trong `starter\.env`, đặt cấu hình kết nối/model và chỉ thêm API key nếu chạy Claude:

```dotenv
DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab
EMBEDDING_MODEL=C:\models\embeddinggemma-2
VECTOR_STRATEGY=A
ANTHROPIC_API_KEY=
```

`DATABASE_URL` phải trỏ tới PostgreSQL có bảng `rag_chunks_768` và Strategy A; `EMBEDDING_MODEL` phải khớp model dùng cho index. Điền API key vào `ANTHROPIC_API_KEY`; không commit `.env` hoặc API key. Câu hỏi benchmark nằm trong `starter\data\converted\questions.json`.

### Kiểm tra kết nối và chạy Claude

Từ gốc Day 7, kiểm tra kết nối tới index rồi chạy benchmark agent. Lệnh benchmark gọi Claude API và có thể phát sinh phí:

```powershell
python -c "from starter.pgvector_retriever import PgVectorRetriever; print(len(PgVectorRetriever().search('Đất chợ dân sinh thuộc nhóm đất nào?')))"
python -m starter.benchmark_agent --questions starter\data\converted\questions.json --split test --output starter\results\agent-test\selected-hybrid-k5-b1600 --retrieval hybrid-rrf --max-searches 2 --top-k 5 --budget 1600
```

Nếu muốn đối chiếu với baseline Claude, chạy thêm lệnh baseline ở cuối báo cáo. Ghi mỗi cấu hình vào thư mục output riêng.

## Đánh giá Claude agent trên test

Chạy Claude Haiku 5.5 với Hybrid RRF, tối đa 2 lượt tìm kiếm/câu. So sánh baseline top-k=3, budget=800 với cấu hình ứng viên top-k=5, budget=1600. Chấm thủ công câu trả lời theo thang 0 (sai/không có căn cứ), 1 (đúng một phần) và 2 (đúng, đầy đủ, có căn cứ); citation được kiểm tra riêng với nội dung chunk.

| Chỉ số | Baseline k3/b800 | Ứng viên k5/b1600 |
|---|---:|---:|
| Câu có gọi retrieval | 8/8 | 8/8 |
| Tổng lượt retrieval | 14 | 12 |
| API calls Claude | 22 | 20 |
| Input / output tokens | 23.466 / 2.652 | 23.974 / 2.811 |
| Median thời gian/câu | 11.983 ms | 8.326 ms |
| Điểm grounded 0 / 1 / 2 (trên 6 câu answerable) | 4 / 1 / 1 | 3 / 0 / 3 |
| Điểm grounded trung bình (thang 0–2) | 0,50 | 1,00 |
| Citation được nội dung hỗ trợ | 1/2 | 5/6 |
| Câu trả lời có citation | 2/2 | 3/3 |
| Abstain đúng trên câu thiếu evidence | 2/2 | 2/2 |
| False abstention trên câu answerable | 4/6 | 3/6 |

Kết quả theo câu:

| ID | Baseline | Ứng viên | Nhận xét thủ công |
|---|---|---|---|
| T01 | Abstain | Abstain | Có thể trả lời nhưng cả hai lần search đều không có kết quả. |
| T02 | Abstain | Abstain | Có thể trả lời nhưng cả hai lần search đều không có kết quả. |
| T03 | Điểm 1 | Điểm 2 | Câu trả lời baseline gần đúng nhưng chunk được trích bị cắt trước điều kiện đầy đủ; cấu hình ứng viên trích đoạn hỗ trợ đầy đủ. |
| T04 | Abstain | Điểm 2 | Cấu hình ứng viên nêu đúng giá trị thừa kế; hai citation hỗ trợ nội dung, citation `data:38:2` không hỗ trợ thêm khẳng định đi kèm. |
| T05 | Abstain | Abstain | Có thể trả lời nhưng cả hai lần search đều không có kết quả. |
| T06 | Điểm 2 | Điểm 2 | Câu trả lời có căn cứ; các citation đã chấm đều hỗ trợ nội dung. |
| T07 | Abstain đúng | Abstain đúng | Câu thiếu evidence. |
| T08 | Abstain đúng | Abstain đúng | Câu thiếu evidence. |

**Giới hạn quan trọng:** T01, T02 và T05 truyền lần lượt filter `section=chuong_V`, `section=chuong_VI` và `section=chuong_IV`; trace của cả hai cấu hình ghi nhận các search này trả `results=[]`. Do đó ba false abstention này bị ảnh hưởng bởi filter/index và không thể quy riêng cho khả năng trả lời của Claude. Nguyên nhân metadata không khớp chưa được xác minh. Không sửa filter dựa trên test này hoặc dùng lại test để chọn cấu hình. Kết quả agent cho thấy tín hiệu cải thiện ở cấu hình ứng viên nhưng chưa đủ để kết luận chất lượng tổng quát.

Hai lượt chạy agent trên test ghi nhận 42 API calls, 47.440 input tokens và 5.463 output tokens cộng lại. Đây là số liệu usage của hai benchmark run, không phải hóa đơn chi phí; chưa quy đổi sang USD và không bao gồm lần gọi thử riêng trước đó không được lưu trace.

## Lệnh tái tạo

Chạy từ thư mục gốc:

```powershell
python -m starter.benchmark_agent --questions starter/data/converted/questions.json --split test --output starter/results/agent-test/baseline-k3-b800 --retrieval hybrid-rrf --max-searches 2 --top-k 3 --budget 800
python -m starter.benchmark_agent --questions starter/data/converted/questions.json --split test --output starter/results/agent-test/selected-hybrid-k5-b1600 --retrieval hybrid-rrf --max-searches 2 --top-k 5 --budget 1600
```

Mỗi thư mục output có `summary.json` và `traces.json`. Hai lệnh trên đều gọi Claude API và có thể phát sinh phí. Thư mục `starter/results/` đang bị Git ignore; báo cáo ghi lại số liệu để đối chiếu.
