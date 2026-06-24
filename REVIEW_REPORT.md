# Code Review Report
**Repository:** ThienTan142/News-Article-Recommendation-System
**Date:** 2026-06-17
**Reviewed by:** Codex CLI

## Executive Summary
Dự án là một prototype Python/Streamlit cho recommendation pipeline: tiền xử lý MIND data, tạo embedding, sinh candidate bằng cosine similarity, rank bằng MLP CTR model, rồi rerank bằng MMR. Codebase nhỏ và dễ đọc, nhưng clean clone hiện chưa đủ để chạy vì các file dữ liệu bắt buộc trong `data/precompute/` không được track hoặc mô tả cách tạo đầy đủ. Rủi ro lớn nhất nằm ở dependency manifest chưa đúng, load artifact bằng `pickle`/`torch.load`, thiếu test, thiếu error handling có cấu trúc, và performance chưa cache các phần nặng. Repo cũng đang commit artifact build/cache như `.pyc`, `.egg-info`, và model binary mà không có provenance.

## Critical Issues 🔴
- **Clean clone không có dữ liệu bắt buộc nên app/CLI/training có thể crash ngay.** `app.py:13-14`, `src/run_recommend_cli.py:11-12`, và `src/train.py:79-83` đều phụ thuộc `data/precompute/news_embeddings.npy`, `data/precompute/news_metadata.csv`, `data/precompute/user_history.pkl`, và `data/precompute/ctr_dataset.csv`, nhưng `.gitignore:3-6` bỏ qua `data/`, `*.csv`, `*.npy`, `*.pkl`. Cần có README hướng dẫn precompute end-to-end, kiểm tra file tồn tại, và thông báo lỗi thân thiện trước khi load.
- **Dependency manifest không cài đủ app chính và có pin sai/nguy hiểm.** `app.py:2` import `streamlit` nhưng `requirements.txt` không khai báo `streamlit`, nên môi trường tạo từ requirements sẽ thiếu entry point web. `requirements.txt:22` pin `requests==2.32.0`, bản này đã bị PyPI yanked do xung đột với mitigation CVE-2024-35195. `requirements.txt:24` pin `selenium==5.11.0`, trong khi PyPI Selenium Python hiện đang ở nhánh 4.x và release history không có 5.11.0, nên install có khả năng fail.
- **Load dữ liệu/model bằng định dạng pickle-based mà không có trust boundary.** `src/user_profile.py:6-7` dùng `pickle.load()` và `src/ranking.py:14` dùng `torch.load()` trên artifact trong workspace. Nếu các file này đến từ nguồn không tin cậy hoặc bị thay thế, đây là rủi ro arbitrary code execution. Cần dùng định dạng an toàn hơn cho user history, pin/checksum artifact, và với PyTorch state dict nên dùng cơ chế load weights-only khi stack hỗ trợ.

## Major Issues 🟠
- **Không có test coverage.** Repo không có `tests/`, không có config pytest, và không có CI. Các logic quan trọng như `get_topk_candidates`, `mmr_rerank`, `build_user_vector_from_history`, `rank_candidates`, data preprocessing, và cold-start path đều chưa có regression tests.
- **README gần như trống.** `README.md:1` chỉ có tên dự án, không mô tả kiến trúc, dataset, lệnh precompute, train, run Streamlit, CLI usage, version Python, hoặc thứ tự tạo artifact.
- **Performance request-time chưa phù hợp nếu dữ liệu lớn.** `src/candidate_generation.py:6-9` tính cosine similarity trên toàn bộ corpus mỗi request; `src/ranking.py:13-15` load model mỗi lần rank; `app.py:13-14` load embedding/history ở module level nhưng không dùng `st.cache_data`/`st.cache_resource`. Với corpus lớn hơn demo, latency và memory pressure sẽ tăng rõ rệt.
- **Training dataset rebuild user vector quá tốn kém.** Trong `src/train.py:50-59`, mỗi sample gọi `build_user_vector_from_history()`, hàm này lại dựng `id2idx` từ toàn bộ metadata ở `src/user_profile.py:16`. Đây là chi phí lặp theo số sample; nên precompute `id2idx` và user vectors một lần.
- **Error handling còn quá rộng hoặc thiếu.** `app.py:32-36` và `src/run_recommend_cli.py:22-26` bắt mọi exception từ ranking rồi fallback similarity, dễ che lỗi model/data thật. `src/train.py:142-145` dùng bare `except`, làm mất nguyên nhân lỗi validation metric. `src/embedding.py:5-8` không kiểm tra missing file, schema, shape mismatch giữa embedding và metadata.
- **Repo đang track artifact không nên nằm trong source control.** `src/__pycache__/*.pyc` và `news_project.egg-info/*` là generated artifacts. `models/ctr_model.pt` là binary model 922 KB nhưng không có versioning/provenance/hash; nếu giữ trong Git cần mô tả rõ nguồn tạo và compatibility.
- **Negative sampling trong CTR dataset có bias.** `src/ctr_dataset.py:17-19` lấy `neg[:neg_ratio]` thay vì random/sample có seed, khiến training phụ thuộc thứ tự impressions và có thể làm lệch phân phối negative samples.
- **Packaging chưa hoàn chỉnh.** `setup.py:3-7` chỉ khai báo name/version/packages, không có `install_requires`, metadata, Python version, console entry points, hoặc mapping đến Streamlit app/CLI.

## Minor Issues 🟡
- **Nhiều dependency có vẻ dư so với imports hiện tại.** Code không thấy dùng trực tiếp `nltk`, `spacy`, `gensim`, `xgboost`, `requests`, `beautifulsoup4`, `selenium`, `matplotlib`, `seaborn`, `plotly`, `python-dotenv`, `torchvision`. Nếu chưa cần, nên loại bỏ để giảm thời gian install và bề mặt supply-chain.
- **Một số import/biến không dùng.** `app.py:3-4` import `numpy`/`pandas` nhưng không dùng trực tiếp; `src/ranking.py:3` import `cosine_similarity` không dùng; `src/train.py:11` import `dump` không dùng; `scripts/precompute_news.py:4` import `tqdm` không dùng trực tiếp.
- **Type/schema contract chưa rõ.** Các hàm giả định `news_meta` có `news_id`, `title`, `text`, embedding rows align với metadata rows, và `user_history` là dict user -> set/list news ids, nhưng không có validation hoặc typing.
- **CLI input chưa validate.** `src/run_recommend_cli.py:36-37` nhận `--user` và `--topk`, nhưng không giới hạn topk dương, không kiểm tra `candidate_k >= topk`, và không có message rõ khi dữ liệu chưa được precompute.
- **Dùng emoji trong training logs.** `src/train.py:79`, `src/train.py:88`, `src/train.py:104`, `src/train.py:125`, `src/train.py:147`, `src/train.py:150` có thể gây lỗi hiển thị ở một số terminal Windows nếu encoding chưa đúng.
- **`.gitignore` quá rộng cho artifact dữ liệu mà pipeline cần.** Việc ignore toàn bộ `*.csv`, `*.npy`, `*.pkl` là hợp lý để tránh commit data lớn, nhưng cần `.env.example`/manifest hoặc docs mô tả nơi tải/tạo dữ liệu.

## Strengths ✅
- Codebase nhỏ, module hóa cơ bản theo các bước pipeline: preprocess, embedding, user profile, candidate generation, ranking, diversity, train, app/CLI.
- Recommendation flow có fallback cold-start bằng mean embedding trong `app.py:24-27` và `src/run_recommend_cli.py:14-17`.
- MMR reranking được tách riêng ở `src/diversity.py`, giúp phần diversity độc lập với ranking model.
- Requirements được pin version thay vì thả nổi hoàn toàn, giúp tái lập môi trường dễ hơn sau khi sửa các pin sai/thiếu.
- Không phát hiện secret/token/password trong tracked text files qua scan pattern; binary files `models/ctr_model.pt` và `.pyc` được ghi nhận nhưng không dump nội dung.

## Recommended Action Plan
1. **Làm repo chạy được từ clean clone.** Thêm README đầy đủ: Python version, setup env, tải MINDsmall, chạy `scripts/precompute_news.py`, chạy `scripts/precompute_user_history.py`, tạo CTR dataset, train model, chạy CLI/Streamlit. Thêm kiểm tra file tồn tại và lỗi hướng dẫn trong `load_news_embeddings()`/`load_user_history()`.
2. **Sửa dependency manifest.** Thêm `streamlit`, bỏ hoặc sửa `selenium==5.11.0`, nâng `requests` khỏi bản yanked, bỏ dependency chưa dùng, và cân nhắc dùng lock file (`requirements.lock`/`pip-tools`/`uv.lock`) cho reproducibility.
3. **Hardening artifact loading.** Thay `pickle` bằng JSON/Parquet/Arrow nếu có thể; nếu vẫn cần pickle/torch artifact, thêm checksum/provenance, giới hạn đường dẫn load, và dùng load weights-only cho model state dict khi phù hợp.
4. **Thêm test trước khi refactor sâu.** Ưu tiên unit tests cho candidate generation, MMR, user profile cold-start, rank fallback, schema/shape mismatch, missing data files, và CTR sample generation.
5. **Tối ưu runtime.** Cache embedding/history/model trong Streamlit, load model một lần, precompute user vectors/id mapping trong training, và cân nhắc ANN index nếu số lượng article lớn.
6. **Dọn repo hygiene.** Remove tracked `__pycache__`, `.egg-info`, và quyết định rõ model binary nên nằm trong Git, Git LFS, release artifact, hay được train lại từ script. Cập nhật `.gitignore` tương ứng.
7. **Cải thiện error handling và observability.** Thay broad/bare exception bằng exception cụ thể, log lỗi kỹ thuật cho developer, hiển thị message thân thiện cho user, và fail fast khi data/model/schema không hợp lệ.

Nguồn kiểm tra dependency bên ngoài: [PyPI requests 2.32.0](https://pypi.org/project/requests/2.32.0/) và [PyPI selenium](https://pypi.org/project/selenium/), truy cập ngày 2026-06-17.
