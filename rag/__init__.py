LLM_FOR_TER = "gemini/gemini-1.5-flash"
LLM_FOR_CHECKING_RAG_KNOWLEDGE = "gemini/gemini-1.5-flash"
LLM_FOR_ANSWERING = "gemini/gemini-2.0-flash-exp"
LLM_FOR_SUMMARIZING = "gemini/gemini-2.0-flash-exp"
# LLM_ADVANCED_REASONING = "gemini/gemini-2.0-flash-thinking-exp-1219"

HUGGINGFACE_EMBEDDING = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
][1]

# https://www.datacamp.com/tutorial/agentic-rag-tutorial
# https://viettel.vn/di-dong/goi-cuoc
# https://chatgpt.com/c/68046a9e-b8ac-8003-853d-7e9448f1483c
# https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

from .base import log_info, log_info_and_print
from typing import Any, TypedDict
from .subscriber import subscribe

class HistoryUpdate(TypedDict):
    """History update type for chat history - see MyRAG::squeeze_history() below"""
    assistant: str
    user: str
    user_first: bool

log_info(f"Importing libraries...")
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from langchain.schema import Document
from .llm import LLM
from .reasoning_data_query_engine import ReasoningDataQueryEngine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import math
from threading import Thread, Lock
from typing import Literal, Callable
import MySQLdb
import asyncio
from .db import fetch_interpretations_and_packages, update_summary_and_satisfaction, fetch_faqs

log_info(f"Importing libraries: Done.")

log_info(f"Setting up environment variables...")
load_dotenv()
log_info(f"Setting up environment variables: Done.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in the .env file."
    )

DATABASE_HOST = os.getenv("READONLY_DATABASE_HOST", "")
DATABASE_PORT_STR = os.getenv("READONLY_DATABASE_PORT", "")
DATABASE_NAME = os.getenv("READONLY_DATABASE_NAME", "")
DATABASE_USER = os.getenv("READONLY_DATABASE_USER", "")
DATABASE_PASSWORD = os.getenv("READONLY_DATABASE_PASSWORD", "")

if not all(
    [DATABASE_HOST, DATABASE_PORT_STR, DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD]
):
    raise ValueError(
        "Some or all of the database environment variables are not set. Please set them in the .env file."
    )
DATABASE_PORT = int(DATABASE_PORT_STR)
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT_STR = os.getenv("REDIS_PORT", "")
if not all([REDIS_HOST, REDIS_PORT_STR]):
    raise ValueError(
        "Some or all of the Redis environment variables are not set. Please set them in the .env file."
    )
REDIS_PORT = int(REDIS_PORT_STR)

SOCKET_HOST = os.getenv("SOCKET_HOST", "")
SOCKET_PORT_STR = os.getenv("SOCKET_PORT", "")
if not all([SOCKET_HOST, SOCKET_PORT_STR]):
    raise ValueError(
        "Some or all of the socket environment variables are not set. Please set them in the .env file."
    )
SOCKET_PORT = int(SOCKET_PORT_STR)

def is_value_present(x: Any):
    """Check if the value is present and not NaN"""
    return (
        x is not None
        and x != ""
        and x != "nan"
        and x != "None"
        and (not isinstance(x, float) or not math.isnan(x))
    )

class MyRAG:
    def __init__(self, db: MySQLdb.Connection):
        log_info("Initializing RAG...")

        self.db = db

        self.llm_for_ter = LLM(
            model=LLM_FOR_TER,
            api_key=GEMINI_API_KEY,
            # max_tokens=500,
            temperature=0.0,
        )

        self.llm_for_answering = LLM(
            model=LLM_FOR_ANSWERING,
            api_key=GEMINI_API_KEY,
            # max_tokens=500,
            temperature=0.1,
        )

        self.llm_for_checking_rag_knowledge = LLM(
            model=LLM_FOR_CHECKING_RAG_KNOWLEDGE,
            api_key=GEMINI_API_KEY,
            # max_tokens=500,
            temperature=0.0,
        )

        self.llm_for_summarizing = LLM(
            model=LLM_FOR_SUMMARIZING,
            api_key=GEMINI_API_KEY,
            # max_tokens=500,
            temperature=0.0,
        )

        self.embeddings = self.setup_embeddings()
        self.interpretations = ""
        self.vector_store = self.setup_faq_vector_store(self.embeddings)

        self.interpretations, self.field_names_to_local_names_map, df = fetch_interpretations_and_packages(self.db)
        self.reasoning_data_q_engine = ReasoningDataQueryEngine(
            df, lambda x: np.array(self.embeddings.embed_query(x))
        )
        log_info("Initializing RAG: Done.")
    
    def check_local_knowledge_faq(self, query: str, chat_history: str, local_content: str):
        """Router function to determine if we can answer from local knowledge"""
        prompt = f"""Vai trò: Bạn là trợ lý ảo thông minh có khả năng giải đáp thắc mắc của người dùng.
        Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho và lịch sử chat hay không. (Đó là tóm tắt lịch sử chat giữa bạn và người dùng)
        Hướng dẫn:
        - Phân tích kiến thức được cho và xác định liệu sử dụng kiến thức đó có giúp ích cho việc trả lời câu hỏi hay không, có liên quan trực tiếp đến câu hỏi hay không.
        - Đưa ra câu trả lời rõ ràng và ngắn gọn, chỉ ra rằng bạn có thể trả lời câu hỏi mà không cần thêm thông tin nào khác.
        - Chú ý một số từ hay được nói tắt: "gói" thay cho "gói cước", "mạng" thay cho "mạng xã hội", "mạng Internet", "dữ liệu" hoặc "nhà mạng Viettel"...
        Định dạng truy vấn đầu vào:
        - Lịch sử chat: ...
        - Kiến thức: ...
        - Câu hỏi: ...
        Định dạng câu trả lời của bạn:
        - Trả lời: Có/Không

        Hãy nghiên cứu các ví dụ phía dưới và dựa vào đó, trả lời truy vấn được đưa ra ở cuối cùng.
        Ví dụ 1:
        - Lịch sử chat: Không có
        - Kiến thức: Viettel là một trong những nhà mạng lớn nhất tại Việt Nam.
        - Câu hỏi: Viettel có phải là nhà mạng lớn nhất tại Việt Nam không?
        - Trả lời: Có
        Ví dụ 2:
        - Lịch sử chat: Không có
        - Kiến thức: Viettel có nhiều gói cước khác nhau cho khách hàng.
        - Câu hỏi: Viettel có gói cước nào tốt nhất không?
        - Trả lời: Không
        Ví dụ 3:
        - Lịch sử chat: Không có
        - Kiến thức: Viettel cung cấp dịch vụ 4G và 5G.
        - Câu hỏi: Viettel có cung cấp dịch vụ 3G không?
        - Trả lời: Không
        Ví dụ 4:
        - Lịch sử chat: Không có
        - Kiến thức: Gói SD70 có giá là 70.000 VNĐ một tháng, tự động gia hạn.
        - Câu hỏi: Giá gói cước SD70 là bao nhiêu?
        - Trả lời: Có
        Ví dụ 5:
        - Lịch sử chat: Không có
        - Kiến thức: Gói SD70 có giá là 70.000 VNĐ một tháng, tự động gia hạn.
        - Câu hỏi: Gói SD70 cung cấp dịch vụ gì?
        - Trả lời: Không
        Ví dụ 6:
        - Lịch sử chat:
            Trợ lý ảo gợi ý gói 6MXH100 (180GB/tháng) và 12MXH100 (360GB/tháng) vì phù hợp nhu cầu xem phim nhiều.
            Người dùng hỏi cụ thể về ưu đãi của các gói 6MXH100 và 12MXH100, muốn tìm gói rẻ mà nhiều data.
            Trợ lý ảo gợi ý thêm các gói SD70 (70.000đ/tháng, 30GB), V90B (90.000đ/tháng, 30GB) và MXH100 (100.000đ/tháng, 30GB), lưu ý data có thể không đủ nếu xem phim nhiều.
            Người dùng muốn được tư vấn gói 70.000đ/tháng.
            Trợ lý ảo xác nhận gói SD70 (70.000đ/tháng, 30GB) phù hợp yêu cầu, nhưng lưu ý 30GB có thể không đủ cho nhu cầu xem phim nhiều. Gợi ý tham khảo các gói data lớn hơn nếu cần.
        - Kiến thức: Gói SD70 có giá là 70.000 VNĐ một tháng, tự động gia hạn. Để đăng ký, soạn tin "SD70 DK8" gửi 290.
        - Câu hỏi: À vậy gói này đăng ký thế nào em nhỉ?
        - Trả lời: Có
        Ví dụ 7:
        - Lịch sử chat:
            Trợ lý ảo gợi ý gói 6MXH100 (180GB/tháng) và 12MXH100 (360GB/tháng) vì phù hợp nhu cầu xem phim nhiều.
            Người dùng hỏi cụ thể về ưu đãi của các gói 6MXH100 và 12MXH100, muốn tìm gói rẻ mà nhiều data.
            Trợ lý ảo gợi ý thêm các gói SD70 (70.000đ/tháng, 30GB), V90B (90.000đ/tháng, 30GB) và MXH100 (100.000đ/tháng, 30GB), lưu ý data có thể không đủ nếu xem phim nhiều.
            Người dùng muốn được tư vấn gói 70.000đ/tháng.
            Trợ lý ảo xác nhận gói SD70 (70.000đ/tháng, 30GB) phù hợp yêu cầu, nhưng lưu ý 30GB có thể không đủ cho nhu cầu xem phim nhiều. Gợi ý tham khảo các gói data lớn hơn nếu cần.
        - Kiến thức: Gói SD70 có giá là 70.000 VNĐ một tháng, tự động gia hạn. Để đăng ký, soạn tin "SD70 DK8" gửi 290.
        - Câu hỏi: À vậy gói 6MXH100 đăng ký thế nào em nhỉ?
        - Trả lời: Không
        
        Truy vấn:
        - Lịch sử chat: {chat_history}
        - Kiến thức: {local_content}
        - Câu hỏi: {query}
        """

        response = self.llm_for_checking_rag_knowledge.call(prompt)
        log_info(f"check_local_knowledge: response: {response}")
        # return response.strip().lower() == "- trả lời: có"
        return "có" in response.strip().lower()

    def write_local_knowledge_reasoning_query(self, chat_history: str, query: str) -> str | None:
        prompt = f"""
        Bạn là một trợ lý ảo thông minh của nhà mạng Viettel, có khả năng giải đáp thắc mắc của người dùng.
        Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho hay không, bằng cách truy vấn từ cơ sở dữ liệu.
        Bạn được cung cấp một cơ sở dữ liệu các gói cước (gọi tắt là gói) của Viettel. Đây là cơ sở dữ liệu dạng bảng, mỗi hàng chứa thông tin của một gói, mỗi cột chứa thuộc tính cụ thể của gói đó.
        Một số hàng có thể trống (optional).
        Các cột của bảng bao gồm:

        {self.interpretations}

        Chú ý 1: Nếu dữ liệu theo ngày là số dương thì nghĩa là một ngày người dùng chỉ được dùng tối đa bấy nhiêu dữ liệu mà thôi, sang ngày khác lại được thêm. Còn nếu không có dữ liệu theo ngày thì nghĩa là người dùng được dùng thoải mái toàn bộ dữ liệu trong chu kỳ mà không bị giới hạn theo ngày, cho đến khi hết dữ liệu trong chu kỳ đó thì phải chờ chu kỳ tiếp theo (nếu gia hạn) mới được tiếp tục sử dụng.
        Chú ý 1b: Nếu người dùng hỏi dung lượng thì cần chọn các cột sau: "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ", "Chi tiết".
        Chú ý 2: Bạn phải luôn SELECT các cột sau trong mọi trường hợp: "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)" và "Chi tiết".
        Chú ý 3: Nếu người dùng nhờ tư vấn cho điện thoại cục gạch, nghe gọi ít hoặc ít sử dụng mạng... thì bạn cần hiểu là phải tìm gói cước rẻ nhất.

        Cú pháp truy cập lấy dữ liệu từ cơ sở dữ liệu như sau:
        SELECT "Tên cột 1", "Tên cột 2"
        WHERE "Tên cột 3" = "Giá trị 3" AND "Tên cột 4" > "Giá trị 4"...
        OR "Tên cột 5" < "Giá trị 5" AND "Tên cột 6" <= "Giá trị 6"...
        OR "Tên cột 7" REACHES MIN
        OR "Tên cột 8" REACHES MAX
        OR "Tên cột 9" CONTAINS "Giá trị 9"...
        ...

        trong đó tên cột và giá trị luôn ở trong dấu nháy (") cho dù đó là giá trị số đi chăng nữa (chẳng hạn "6").
        Tên cột cũng như giá trị sẽ không bao giờ và không được chứa một dấu nháy khác trong đó, nếu không truy vấn sẽ bị coi là sai. Ví dụ "6"" là sai.
        Bạn không cần viết những điều kiện loại bỏ dữ liệu sai ví dụ "4G tốc độ cao/chu kỳ" > 0. Hãy mặc định dữ liệu luôn đúng.
        Bạn không được phép dùng dấu ngoặc đơn như này ( hoặc như này ) để nhóm các biểu thức logic AND-OR. Hãy cố gắng "phá ngoặc" để viết lại câu truy vấn cho dễ hiểu hơn nhé.
        Thứ tự ưu tiên luôn là AND trước rồi mới đến OR.
        Bạn cũng không được phép dùng các toán tử so sánh khác ngoài =, >, <, >=, <=, REACHES MIN, REACHES MAX, CONTAINS.
        Bạn cũng không được phép dùng các toán tử logic khác ngoài AND, OR.
        Bạn cũng không được phép dùng các toán tử khác ngoài SELECT, WHERE.
        Mệnh đề WHERE là bắt buộc.
        Khi người dùng hỏi giá rẻ, giá rẻ nhất thì cần viết query theo kiểu "Giá (VNĐ)" REACHES MIN, chứ không được so sánh với một giá trị cụ thể nào đó, chẳng hạn "Giá (VNĐ)" < 100000.
        Tuy nhiên nếu người dùng hỏi "giá rẻ hơn" thì phải dựa vào lịch sử chat để biết người dùng đang nói tới những gói nào, sau đó xác định gói rẻ hơn trong các gói đó.
        Nếu người dùng hỏi "nhiều data", "data không giới hạn", "miễn phí"... thì nên chọn các gói có "4G tốc độ tiêu chuẩn/ngày" REACHES MIN hoặc "4G tốc độ cao/ngày" REACHES MAX, hoặc cột "Chi tiết" CONTAINS "không giới hạn", "thả ga", "miễn phí" .v.v.

        Trong trường hợp bạn có thể trả lời câu hỏi của người dùng bằng cách tạo một truy vấn dữ liệu như trên, hãy trả về truy vấn.
        Nếu không tạo được truy vấn nhưng câu hỏi vẫn thuộc phạm vi thông tin gói cước, sim thẻ, nhà mạng, giá cả... thì trả về:
            SELECT "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)" và "Chi tiết" WHERE "Chi tiết" CONTAINS "<các từ khóa trong câu hỏi của người dùng>"
        Nếu câu hỏi hoàn toàn nằm ngoài phạm vi những thông tin gói cước, sim thẻ... như trên thì trả về IMPOSSIBLE.

        Hãy nghiên cứu các ví dụ dưới đây, và trả lời câu hỏi được đưa ra ở cuối cùng:
        Ví dụ 1:
        - Lịch sử chat: Không có
        - Câu hỏi: Gói cước nào có giá rẻ nhất?
        - Trả lời: SELECT "Mã dịch vụ", "Giá (VNĐ)" WHERE "Giá (VNĐ)" REACHES MIN
        Ví dụ 2:
        - Lịch sử chat: Không có
        - Câu hỏi: Làm thế nào để đăng ký dịch vụ SD70?
        - Trả lời: SELECT "Chi tiết", "Cú pháp" và "Mã dịch vụ" WHERE "Mã dịch vụ" = "SD70"
        Ví dụ 3:
        - Lịch sử chat: Không có
        - Câu hỏi: Em ơi thế sao thuê bao của anh cứ tự trừ tiền thế nhỉ, em xem giúp anh số dư còn bao nhiêu với
        - Trả lời: IMPOSSIBLE
        Ví dụ 4:
        - Lịch sử chat: Không có
        - Câu hỏi: Ừ thế xem giúp anh gói nào để anh lướt mạng thả ga đi, một ngày xem phim đã tốn mấy gigabyte rồi
        - Trả lời: SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thả ga" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN
        Ví dụ 5:
        - Lịch sử chat: Không có
        - Câu hỏi: À em ơi bên em có gói nào rẻ mà lướt mạng thoải mái không, chứ một ngày anh lướt mạng hết mấy gigabyte rồi
        - Trả lời: SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày" WHERE "Chi tiết" CONTAINS "lướt mạng thoải mái" AND "4G tốc độ tiêu chuẩn/ngày" REACHES MIN AND "Giá (VNĐ)" REACHES MIN
        Ví dụ 6:
        - Lịch sử chat:
            Trợ lý ảo gợi ý gói 6MXH100 (180GB/tháng) và 12MXH100 (360GB/tháng) vì phù hợp nhu cầu xem phim nhiều.
            Người dùng hỏi cụ thể về ưu đãi của các gói 6MXH100 và 12MXH100, muốn tìm gói rẻ mà nhiều data.
            Trợ lý ảo gợi ý thêm các gói SD70 (70.000đ/tháng, 30GB), V90B (90.000đ/tháng, 30GB) và MXH100 (100.000đ/tháng, 30GB), lưu ý data có thể không đủ nếu xem phim nhiều.
            Người dùng muốn được tư vấn gói 70.000đ/tháng.
            Trợ lý ảo xác nhận gói SD70 (70.000đ/tháng, 30GB) phù hợp yêu cầu, nhưng lưu ý 30GB có thể không đủ cho nhu cầu xem phim nhiều. Gợi ý tham khảo các gói data lớn hơn nếu cần.
        - Câu hỏi: À vậy gói này đăng ký thế nào em nhỉ?
        - Trả lời: SELECT "Mã dịch vụ", "Cú pháp", "Giá (VNĐ)", "Chi tiết" WHERE "Mã dịch vụ" = "SD70"

        Lịch sử chat: {chat_history}
        Câu hỏi: {query}
        """
        log_info(f"write_local_knowledge_reasoning_query: prompt: {prompt}")
        log_info(f"write_local_knowledge_reasoning_query: calling...")
        response = self.llm_for_checking_rag_knowledge.call(prompt)
        log_info(f"write_local_knowledge_reasoning_query: response: {response}")
        return None if "impossible" in response.strip().lower() else response.strip()

    def check_local_knowledge_reasoning(self, chat_history: str, query: str):
        q = self.write_local_knowledge_reasoning_query(chat_history, query)
        if q is None:
            return None
        log_info(f"check_local_knowledge_reasoning: query written by LLM: {q}")
        try:
            queryObject = ReasoningDataQueryEngine.compile(q)
        except Exception as e:
            log_info_and_print(f"check_local_knowledge_reasoning: query compilation failed: {e}")
            log_info(f"Initial query written by LLM: {q}")
            return None
        else:
            queryObjectInterpretation = queryObject.interpret(entity_name="gói cước")
            log_info(f"check_local_knowledge_reasoning: compiled query: {queryObjectInterpretation}")
            table = self.reasoning_data_q_engine.apply(queryObject)
            log_info(f"check_local_knowledge_reasoning: applied table: {table}")
            context = f"""
            Đã xác định được ngữ cảnh phù hợp. Các gói cước mà người dùng mong muốn là: {queryObjectInterpretation}

            Cụ thể, các gói cước đó là:
            {self.combine_documents(
                self.convert_df_to_documents(table.to_df())
            )}

            Hãy trả lời người dùng dựa trên ngữ cảnh này nhé. Đây là tất cả các gói cước phù hợp với yêu cầu của người dùng. Không còn gói cước nào khác.
            Vì vậy không được nói: "Em xin lỗi vì hiện tại em chưa thể cung cấp thêm thông tin về các gói cước khác phù hợp hơn với nhu cầu nghe gọi của anh/chị..."
            """
            return context

    def setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING,
            model_kwargs={"device": "cuda"},
        )

    def setup_faq_vector_store(self, embeddings):
        """Setup FAQ vector store from MySQL database"""
        log_info("Setting up vector store...")

        log_info("Setting up vector store: Loading documents...")

        df = fetch_faqs(self.db)
        df.loc[len(df)] = ["Viettel là gì?", "Viettel là tập đoàn hàng đầu trong lĩnh vực viễn thông tại Việt Nam."]  # avoid list index out of range error
        documents = self.convert_df_to_documents(df)

        # Split the documents into smaller chunks
        log_info("Setting up vector store: Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n---\n"],
        )
        splits = text_splitter.split_documents(documents)

        # Create the vector store
        log_info("Setting up vector store: Creating FAISS index...")
        vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store

    def get_local_content(self, vector_store, query):
        """Get local content from vector store"""
        log_info(f"Getting the most relevant local content...")
        k = 5

        # Get the top k most relevant documents
        log_info(
            f"Getting the most relevant local content: Performing similarity search (k = {k})..."
        )
        docs = vector_store.similarity_search(query, k=k)

        # Combine the content of the documents into a single string
        log_info(f"Getting the most relevant local content: Combining documents...")
        context = self.combine_documents(docs)

        log_info(
            f"Getting the most relevant local content: Combined context: {context}"
        )
        return context

    def combine_documents(self, docs):
        return "\n---\n".join([doc.page_content for doc in docs])

    def convert_df_to_documents(self, df: pd.DataFrame):
        # Convert the DataFrame to a list of Document objects
        documents = []
        for _, row in df.iterrows():  # type: ignore
            row_data = list(
                (key, value)
                for (key, value) in row.items()
                if is_value_present(value)
            )
            content = (
                "\n".join((f"{key}: {value}" for key, value in row_data))
                + "\n---\n"
            )
            doc = Document(page_content=content, metadata=dict(row_data))
            documents.append(doc)

        return documents

    def get_answer_from_local_knowledge(self, vector_store, chat_history, query):
        """Get answer from local knowledge (FAQ)"""
        context = self.get_local_content(vector_store, query)
        if self.check_local_knowledge_faq(query, chat_history, context):
            return context
        else:
            return None

    def process_query(self, chat_history, query: str, customer_emotion: str):
        log_info(f"PROCESSING QUERY: {query}")

        context = None

        # Check reasoning-based RAG
        reasoning_retrieval_start_time = pd.Timestamp.now()
        print(f"Checking reasoning-based RAG...")
        reasoning_data_context = self.check_local_knowledge_reasoning(chat_history, query)
        print(f"Checking reasoning-based RAG: Done.")
        reasoning_retrieval_end_time = pd.Timestamp.now()
        if reasoning_data_context is not None:
            context = reasoning_data_context
            retrieval_time = reasoning_retrieval_end_time - reasoning_retrieval_start_time
        else:
            log_info(f"Cannot answer by reasoning RAG, resorting to relevance-based RAG.")

        if context is None:
            # Check relevance-based RAG
            retrieval_start_time = pd.Timestamp.now()
            local_content = self.get_local_content(self.vector_store, query)
            can_answer_locally = self.check_local_knowledge_faq(query, chat_history, local_content)
            retrieval_end_time = pd.Timestamp.now()
            retrieval_time = retrieval_end_time - retrieval_start_time
            if can_answer_locally:
                context = f"""Đã xác định được ngữ cảnh phù hợp. Các gói cước mà người dùng mong muốn là:
                
                {local_content}

                Hãy trả lời người dùng dựa trên ngữ cảnh này nhé."""

        if context is not None:
            # Get context from local vector store
            log_info(f"Context generated from local knowledge (relevance-based).")
        else:
            log_info(f"Cannot answer locally, resorting to reasoning LLM.")
            context = f"""
            Hệ thống chưa thể xác định ngữ cảnh liên quan.
            Nếu câu hỏi liên quan đến thông tin thực tế cần kiểm chứng (ví dụ: thông tin gói cước, chương trình khuyến mãi, thời gian sử dụng, thông tin tài khoản, v.v.), thì ngữ cảnh này chưa xác định được.
            Do vậy, bạn không được phép bịa ra những thông tin nằm ngoài ngữ cảnh và lịch sử trò chuyện, chẳng hạn bịa ra những gói cước không có thật, nói sai dung lượng, giá cả... của gói cước.
            Bạn cần phản hồi đại loại như sau:
            Xin lỗi, em không thể trả lời câu hỏi này. Xin quý khách đợi giây lát để được nối máy tới nhân viên hỗ trợ kỹ thuật chuyên nghiệp.
            Ngược lại, nếu câu hỏi mang tính tổng quát, suy luận hoặc không yêu cầu dữ liệu cụ thể về nhà mạng hay các thông tin cần kiểm chứng khác, bạn có thể trả lời dựa trên kiến thức tổng hợp sẵn có của bạn, một cách rõ ràng, lịch sự và hữu ích.
            """

        log_info(f"Context: {context}")

        # Generate final answer
        return (
            self.generate_final_answer(chat_history, context, query, customer_emotion),
            retrieval_time,
        )

    def generate_final_answer(self, chat_history: str, context: str, query: str, customer_emotion: str):
        """Generate final answer using LLM"""
        prompt = f"""Bạn là một trợ lý ảo thông minh, là nhân viên chăm sóc khách hàng của Viettel. Bạn có khả năng trả lời câu hỏi của người dùng dựa trên ngữ cảnh đã cho.
            Bạn có thể sử dụng ngữ cảnh đã cho để trả lời câu hỏi của người dùng một cách chính xác và nhanh chóng.
            Hãy đảm bảo rằng câu trả lời của bạn là chính xác và đầy đủ, dựa trên ngữ cảnh đã cho, cũng như lịch sử trò chuyện. Bạn cần xử lý linh hoạt đặc biệt là các từ đồng nghĩa.
            Lưu ý không được sử dụng Markdown, chẳng hạn không được in đậm chữ như kiểu **này**.
            Bạn cũng cần thêm những từ ngữ hòa nhã, lịch sự, thân thiện, thể hiện sự chuyên nghiệp trong câu trả lời của mình, chẳng hạn "nhé", "Xin chào quý khách", "Dạ, bên em đã nhận được câu hỏi của anh/chị"...
            Hãy xưng "em" và gọi người dùng là "anh" hoặc "quý khách".
            Chú ý rằng ngữ cảnh là do hệ thống tự sinh ra dựa trên chính câu nói của người dùng, và có thể không chính xác hoàn toàn.
            Hãy tưởng tượng ngữ cảnh là do chính bản thân bạn tìm kiếm, tra cứu mà có. Vì vậy không được nói "hiện tại hệ thống đang tìm kiếm gói cước A, B" mà nên nói "Dạ, em vừa tìm được gói cước A, B có thể đáp ứng nhu cầu của anh/chị ạ"...
            Hãy trả lời người dùng một cách tự nhiên, như thể bạn đang nói chuyện với họ.
            Khi tư vấn gói dữ liệu cần linh hoạt trong ước lượng lượng dữ liệu phù hợp nhu cầu người dùng, tránh cứng nhắc, không cố tìm con số chính xác. Tìm con số gần nhất và tư vấn cho khách gói cước phù hợp nhất.
            Trong mọi trường hợp, bạn KHÔNG được hướng dẫn người dùng liên hệ tổng đài chăm sóc khách hàng của Viettel, hoặc truy cập website của Viettel.
            Bạn cũng cần chú ý tới cảm xúc của người dùng, nếu người dùng có cảm xúc "tiêu cực" thì bạn cần thể hiện sự đồng cảm và sẵn sàng hỗ trợ họ, nếu cần có thể nói lời xin lỗi, thông cảm.
            Với giá tiền, bạn ghi rõ "Việt Nam Đồng" thay vì viết tắt, ví dụ không viết tắt là "10000 VNĐ", "10000đ" mà phải viết là "Việt Nam Đồng".
            Hãy trả lời ngắn gọn súc tích, không quá dài.

            Dưới đây là tóm tắt lịch sử trò chuyện giữa bạn (Trợ lý ảo) và người dùng (Câu nói của người dùng):
            {chat_history}

            Còn đây là câu hỏi của người dùng:
            {query}

            Cảm xúc của người dùng được thể hiện trong câu trên:
            {customer_emotion}

            Chú ý nếu người dùng nói một gói nào đó "ổn", "tạm được", "được", "ok", "được rồi" thì bạn cần hiểu là người dùng đã đồng ý với gói cước đó, và bạn cần tư vấn thêm về cách đăng ký, chi tiết, giá tiền... gói cước đó, dựa theo ngữ cảnh được cung cấp.
            Ngữ cảnh như sau:
            {context}
        """,
        # TODO: Bạn cũng chú ý giới tính của người dùng.

        return self.llm_for_checking_rag_knowledge.call_tokenstream(prompt)

    def squeeze_history(self, chat_id: int, old_chat_history: str, update: HistoryUpdate) -> str:
        """Summarizes history so that future queries won't overflow the context window."""
        squeeze_history_time_start = pd.Timestamp.now()

        updates_text = ""
        have_assistant_answer = update['assistant'] != ""
        have_user_query = update['user'] != ""
        if have_assistant_answer and not have_user_query:
            updates_text += f"Bây giờ, đoạn hội thoại đã được cập nhật thêm một câu trả lời của trợ lý ảo:\n{update['assistant']}\n"
        elif have_user_query and not have_assistant_answer:
            updates_text += f"Bây giờ, đoạn hội thoại đã được cập nhật thêm một câu nói của người dùng:\n{update['user']}\n"
        elif have_assistant_answer and have_user_query:
            if update['user_first']:
                updates_text += f"""
                    Bây giờ, đoạn hội thoại đã được cập nhật thêm một câu hỏi (hoặc trả lời) của người dùng:
                    {update['user']}

                    Sau đó trợ lý ảo đã trả lời như sau:
                    {update['assistant']}
                """
            else:
                updates_text += f"""
                    Bây giờ, đoạn hội thoại đã được cập nhật thêm một câu trả lời của trợ lý ảo:
                    {update['assistant']}

                    Sau đó người dùng đã hỏi (hoặc trả lời) như sau:
                    {update['user']}
                """
        else:
            updates_text += ""

        try:
            prompt = f"""
            Bạn là một thư ký có khả năng tóm tắt lịch sử trò chuyện giữa người dùng và nhân viên chăm sóc khách hàng của Viettel,
            cũng như có khả năng phân tích mức độ hài lòng của khách hàng.
            Bạn có thể tóm tắt lịch sử trò chuyện một cách ngắn gọn và súc tích, chỉ giữ lại những thông tin quan trọng nhất.
            Đặc biệt, bạn phải chỉ ra được người dùng đang nói tới gói cước nào (nếu có), và những thông tin quan trọng liên quan đến gói cước đó (nếu có).
            Ví dụ người dùng bảo gói này ổn, thì phải xem trước đó họ nói về gói nào, chỉ ra cụ thể tên gói đó.
            Không được nói "gói đã nêu", "gói mà người dùng thấy ổn"... mà phải nói rõ ví dụ gói SD70.

            Về việc đánh giá mức độ hài lòng của khách hàng, bạn đánh giá từ 1 đến 5 sao, trong đó 1 sao là không hài lòng và 5 sao là rất hài lòng.
            Lịch sử trò chuyện trước đây được tóm tắt như sau:
            {old_chat_history}

            {updates_text}

            Bạn hãy tiếp tục tóm tắt lịch sử trò chuyện này một cách ngắn gọn và súc tích, chỉ giữ lại những thông tin quan trọng nhất.
            Hãy chú ý rằng bạn không được phép thay đổi nội dung của lịch sử trò chuyện trước đó, mà chỉ được phép tóm tắt lại nó.
            Bạn cũng không được phép thêm bất kỳ thông tin nào khác ngoài những gì đã có trong lịch sử trò chuyện trước đó.
            Định dạng câu trả lời của bạn:
            Dòng thứ nhất: Mức độ hài lòng, ví dụ: 4 sao
            Từ dòng thứ hai trở đi: Tóm tắt lịch sử trò chuyện, không dùng Markdown, không dùng in đậm chữ như kiểu **này**.

            Ví dụ:
            1 sao
            Khách hàng không hài lòng với gói cước SD70, vì không đủ dung lượng cho nhu cầu sử dụng của họ.
            """
            res = self.llm_for_summarizing.call(prompt)

            parts = res.split('\n', 1)
            first_row = parts[0] if len(parts) > 0 else ""
            the_rest = parts[1] if len(parts) > 1 else ""

            customer_satisfaction = first_row
            chat_history = the_rest

            log_info(f"squeeze_history(): history updated: {chat_history}")
            log_info(f"squeeze_history(): customer satisfaction: {customer_satisfaction}")
            update_summary_and_satisfaction(self.db, chat_id, chat_history, customer_satisfaction)
            return chat_history
        finally:
            squeeze_history_time_end = pd.Timestamp.now()
            log_info_and_print(f"*** Squeezed history in {(squeeze_history_time_end - squeeze_history_time_start).total_seconds()}.")

    def squeeze_history_in_background(self, chat_id: int, old_chat_history: str, update: HistoryUpdate):
        thread = Thread(
            target=self.squeeze_history,
            args=(chat_id, old_chat_history, update,),
        )
        thread.start()

    def answer(self, chat_id: int, chat_history: str, query: str, customer_emotion: str, on_new_token: Callable[[str | None], Any]):
        time_start = pd.Timestamp.now()
        answer_iter, rag_time = self.process_query(chat_history, query, customer_emotion)
        time_end = pd.Timestamp.now()

        print(f"Answer: ", end="")
        full_answer = ""
        for answer in answer_iter:
            if answer is None:
                break
            full_answer += answer
            print(answer, end="", flush=True)
            on_new_token(answer)
        on_new_token(None)
        self.squeeze_history_in_background(chat_id, old_chat_history=chat_history, update=HistoryUpdate(assistant=full_answer, user=query, user_first=True))
        print("\n")
        log_info_and_print(f"*** Took {(time_end - time_start).total_seconds()} seconds of which RAG took {rag_time.total_seconds()} seconds.\n")

class Main:
    def restart_rag(self):
        log_info("Restarting RAG...")
        with self.db_lock:
            if self.db:
                self.db.close()
            self.db = MySQLdb.connect(
                host=DATABASE_HOST,
                user=DATABASE_USER,
                passwd=DATABASE_PASSWORD,
                db=DATABASE_NAME,
                port=DATABASE_PORT,
            )

            with self.rag_lock:
                self._rag = MyRAG(self.db)
        log_info("RAG restarted successfully.")

    def __init__(self):
        self.db_lock = Lock()
        self.rag_lock = Lock()
        self.db = None
        self.restart_rag()
        self.r = subscribe(REDIS_HOST, REDIS_PORT, lambda e: self.restart_rag())
    
    @property
    def rag(self):
        with self.rag_lock:
            return self._rag
