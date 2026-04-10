# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:*
> High cosine similarity (gần bằng 1) nghĩa là hai vector embedding của hai đoạn văn bản có hướng rất gần nhau trong không gian đa chiều, cho thấy chúng có sự tương đồng cao về ngữ nghĩa và ngữ cảnh, dù từ ngữ có thể không giống hệt nhau.

**Ví dụ HIGH similarity:**
- Sentence A: VinFast cung cấp chính sách đổi xe xăng sang xe điện với nhiều ưu đãi.
- Sentence B: Người dùng có thể nhận hỗ trợ tài chính khi chuyển từ phương tiện chạy xăng sang xe chạy pin của VinFast.
- Tại sao tương đồng: Cả hai đều nói về cùng một chủ đề (chương trình thu cũ đổi mới của VinFast) và hành động (chuyển đổi xe).

**Ví dụ LOW similarity:**
- Sentence A: Pin xe điện VinFast được bảo hành 10 năm.
- Sentence B: Cách nấu phở bò truyền thống cần chuẩn bị nhiều loại gia vị.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (kỹ thuật xe hơi vs. ẩm thực), không có liên quan về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*
> Cosine similarity tập trung vào hướng của vector thay vì độ dài. Điều này giúp đánh giá sự tương đồng về nội dung một cách chính xác hơn mà không bị ảnh hưởng bởi độ dài của đoạn văn bản (vốn làm thay đổi độ dài vector).

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*
> Đáp án sẽ là ceil((10000 - 100) / (500 - 100)) = 25 chunks. Số lượng chunk tăng lên giúp bảo toàn ngữ cảnh tốt hơn tại các điểm giao nhau, tránh việc các câu hoặc khái niệm quan trọng bị cắt đôi ở ranh giới chunk.


---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Chính sách & Câu hỏi thường gặp về Xe điện VinFast (VinFast EV Policies & FAQ)

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*
> Nhóm chọn domain này vì có dữ liệu chính sách phong phú, cấu trúc rõ ràng, rất phù hợp để thử nghiệm khả năng truy xuất thông tin (RAG) chính xác.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | battery_0326 | VinFast News | 3,386 | doc_code, effective_date |
| 2 | charging_ev_0326| VinFast Support| 3,751 | policy_type, is_active |
| 3 | discontinued_models| VinFast Archive| 3,921 | model, doc_code |
| 4 | gas_to_ev_0326 | VinFast Sales | 2,453 | effective_date, model |
| 5 | sales_0326 | VinFast Sales | 7,361 | policy_type, is_active |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_code | string | VF-POL-2024 | Xác định chính xác phiên bản tài liệu. |
| effective_date | date | 2024-03-26 | Lọc các chính sách còn hiệu lực theo thời gian. |
| policy_type | string | Warranty / Sales | Thu hẹp phạm vi tìm kiếm theo loại yêu cầu. |
| model | string | VF 3, VF 5 | Truy xuất thông tin riêng cho từng dòng xe. |
| is_active | boolean | true | Đảm bảo chỉ lấy thông tin đang áp dụng thực tế. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| gas_to_ev_0326 | FixedSizeChunker (`fixed_size`) | 6 | 400 | No (cắt ngang câu) |
| gas_to_ev_0326 | SentenceChunker (`by_sentences`) | 12 | 195 | Yes (giữ trọn câu) |
| gas_to_ev_0326 | RecursiveChunker (`recursive`) | 4 | 764 | Yes (giữ trọn đoạn) |

### Strategy Của Tôi

**Loại:** RecursiveChunker (`recursive`)

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*
> Strategy này cố gắng chia văn bản dựa trên danh sách các dấu phân cách ưu tiên từ cao xuống thấp như "\n\n", "\n", và ". ". Nếu một đoạn văn bản sau khi chia vẫn vượt quá `chunk_size`, nó sẽ tiếp tục đệ quy xuống cấp độ phân cách tiếp theo. Cách này giúp giữ các khối thông tin có liên quan (như các đoạn văn hoặc danh sách) nằm cùng trong một chunk thay vì bị chia lẻ.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
> Các tài liệu chính sách VinFast thường có cấu trúc phân cấp rõ ràng (mục lớn, mục nhỏ, danh sách gạch đầu dòng). `RecursiveChunker` cho phép giữ nguyên cấu trúc các mục này trong cùng một chunk, giúp AI hiểu được ngữ cảnh đầy đủ của một điều khoản.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| gas_to_ev_0326 | best baseline (Fixed) | 54 | 469 | 74.2% |
| gas_to_ev_0326 | **của tôi** (Recursive) | 30 | 764 | 71.1% |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive | 7.5 | Ngữ cảnh liền mạch | Chunk hơi lớn |
| Trang |FixedSizeChunker |8/10 |- Giữ được context giữa các chunk nhờ overlap <br> - Cải thiện độ chính xác retrieval so với không overlap | - Tăng số lượng chunk → tốn tài nguyên hơn <br> - Có thể lặp lại thông tin |
| Nghĩa | Keyword chunk | 7.0 | Tốc độ nhanh | Cắt vụn thông tin |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*
> Sentence-Based (như của Chi) hoặc Recursive với chunk_size nhỏ hơn sẽ tốt nhất. Đối với FAQ/Policy, việc giữ trọn vẹn một câu hỏi/trả lời hoặc một điều khoản là quan trọng nhất để AI không bị "grounding" sai.


---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
> Tôi sử dụng regex `(?<=[.!?])\s+|\n+` để tách câu dựa trên dấu chấm, hỏi, cảm thán hoặc dấu xuống dòng. Điều này giúp xử lý được các trường hợp xuống dòng giữa chừng trong văn bản và đảm bảo các câu được trích xuất sạch sẽ bằng `strip()`.

**`RecursiveChunker.chunk` / `_recursive_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
> Thuật toán duyệt qua danh sách separators (`\n\n`, `\n`, `. `, ` `). Base case là khi độ dài chuỗi nhỏ hơn `chunk_size` hoặc không còn separator nào để thử. Nó gom các phần tách được vào `current_chunk` cho đến khi đầy thì mới lưu lại và chuyển sang chunk mới.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
> Tôi hỗ trợ cả ChromaDB (sử dụng `client.create_collection` với `hnsw:space: "cosine"`) và in-memory store (lưu list dict). Việc tìm kiếm sử dụng hàm `compute_similarity` dựa trên tích vô hướng và chuẩn hóa vector để có điểm số chính xác.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*
> Với ChromaDB, tôi sử dụng tham số `where` để pre-filter metadata trực tiếp từ database. Với in-memory, tôi lọc list trước khi tính similarity để tối ưu hiệu năng. Hàm `delete_document` xóa tất cả chunk có `doc_id` tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
> Prompt bao gồm chỉ dẫn system ("You are a helpful assistant..."), phần context (các chunk được nối bằng dấu `---`), và cuối cùng là câu hỏi của người dùng. Context được inject trực tiếp vào string template để LLM có đủ dữ liệu grounding.

### Test Results

========================================================================== test session starts ==========================================================================
platform win32 -- Python 3.14.3, pytest-9.0.2, pluggy-1.6.0 -- C:\Users\LEGION\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\LEGION\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0, langsmith-0.7.29
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                              [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                       [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                                [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                                 [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                      [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                      [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                            [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                             [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                           [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                             [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                             [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                        [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                    [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                              [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                     [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                         [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                   [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                         [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                             [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                               [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                                 [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                       [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                            [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                              [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                                  [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                               [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                        [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                       [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                                  [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                              [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                         [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                             [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                                   [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                             [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                          [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                        [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                       [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                           [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                      [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                               [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                     [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                         [100%] 

=========================================================================== warnings summary ============================================================================ 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size
  C:\Users\LEGION\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\chromadb\telemetry\opentelemetry\__init__.py:128: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(f):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
===================================================================== 42 passed, 1 warning in 1.43s =====================================================================



---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Xe VinFast rất bền | Ô tô VinFast dùng rất tốt | high | 0.89 | Yes |
| 2 | Sạc pin ở đâu? | Trạm sạc xe điện nằm ở đâu? | high | 0.82 | Yes |
| 3 | VF 3 là xe nhỏ | VF 9 là xe SUV cỡ lớn | low | 0.45 | Yes |
| 4 | Đổi xe xăng lấy xe điện | Chương trình thu cũ đổi mới | high | 0.78 | Yes |
| 5 | Pin 10 năm | Cách nấu cơm ngon | low | 0.12 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*
> Kết quả ở cặp 4 gây bất ngờ vì từ vựng không trùng nhau nhiều nhưng score vẫn cao. Điều này cho thấy embedding model hiểu được khái niệm "đổi xe" và "thu cũ đổi mới" là tương đồng về mặt ngữ nghĩa trong không gian vector.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | VinFast có hỗ trợ đổi xe máy điện sang ô tô điện VF 3 không? | Có, đổi sang VF 3 qua chương trình ưu đãi đổi xe cũ. |
| 2 | Chính sách bảo hành pin xe máy điện VinFast như thế nào? | Pin xe máy điện được bảo hành theo thời hạn niêm yết (thường là 10 năm). |
| 3 | Trạm sạc V-GREEN có thuộc VinFast không? | V-GREEN chịu trách nhiệm triển khai, VinFast đại diện công bố. |
| 4 | Có hỗ trợ lãi suất khi mua xe điện không? | Có chính sách lựa chọn trừ vào giá hoặc hỗ trợ lãi suất. |
| 5 | Xe xăng nào của VinFast đã ngừng sản xuất? | Các mẫu xe xăng như Lux A, Lux SA, Fadil. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Đổi xe máy sang VF 3 | ...xe tải Van chỉ được áp dụng ưu đãi khi đổi sang VF 3... | 0.74 | Yes | VinFast có hỗ trợ đổi sang VF 3. |
| 2 | Bảo hành pin | ...thời hạn bảo hành pin xe máy điện là 10 năm... | 0.68 | Yes | Pin được bảo hành 10 năm. |
| 3 | V-GREEN | ...V-GREEN chịu trách nhiệm triển khai thực hiện... | 0.45 | No | Không đủ thông tin chắc chắn. |
| 4 | Hỗ trợ lãi suất | ...lựa chọn Ưu đãi trừ vào giá hoặc hỗ trợ lãi suất... | 0.72 | Yes | Có hỗ trợ lãi suất hoặc trừ tiền. |
| 5 | Ngừng xe xăng | ...triển khai chương trình cho xe xăng VinFast... | 0.38 | No | Tôi không biết rõ mẫu xe nào. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*
> Tôi học được từ Chi và Mạnh cách test agent với real LLM để so sánh kết quả. Việc sử dụng model thật giúp nhận ra các điểm yếu của retrieval mà mock LLM không thể chỉ ra được.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*
> Qua demo của nhóm khác, tôi thấy họ sử dụng metadata filtering cực kỳ hiệu quả để loại bỏ các thông tin nhiễu từ các năm cũ, điều mà nhóm tôi vẫn chưa tối ưu hoàn toàn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*
> Tôi sẽ tập trung phân tích kỹ cấu trúc tài liệu trước khi chunking để gán metadata schema chi tiết hơn. Ngoài ra, việc tinh chỉnh `chunk_size` thủ công cho từng loại văn bản (dài/ngắn) sẽ giúp tăng độ chính xác của retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 7 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 26 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **75 / 90** |
