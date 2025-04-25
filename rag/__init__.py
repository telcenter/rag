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
from typing import Literal

log_info(f"Importing libraries: Done.")

log_info(f"Setting up environment variables...")
load_dotenv()
log_info(f"Setting up environment variables: Done.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it in the .env file."
    )


def is_value_present(x: Any):
    """Check if the value is present and not NaN"""
    return (
        x is not None
        and x != ""
        and x != "nan"
        and x != "None"
        and (not isinstance(x, float) or not math.isnan(x))
    )

# TODO: Build FAQ

class MyRAG:
    def __init__(self):
        log_info("Initializing RAG...")

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
        self.vector_store = self.setup_vector_store(self.embeddings)
        self.reasoning_data_q_engine = ReasoningDataQueryEngine(
            "viettel.csv", lambda x: np.array(self.embeddings.embed_query(x))
        )
        log_info("Initializing RAG: Done.")
        self._chat_history = ""
        self._chat_history_lock = Lock()
    
    @property
    def chat_history(self):
        with self._chat_history_lock:
            x = self._chat_history
            if not x:
                return "Không có"
            return x

    def check_local_knowledge(self, query: str, local_content: str):
        """Router function to determine if we can answer from local knowledge"""
        prompt = f"""Vai trò: Bạn là trợ lý ảo thông minh có khả năng giải đáp thắc mắc của người dùng.
        Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho và lịch sử chat hay không. (Đó là tóm tắt lịch sử chat giữa bạn và người dùng)
        Hướng dẫn:
        - Phân tích kiến thức được cho và xác định liệu sử dụng kiến thức đó có giúp ích cho việc trả lời câu hỏi hay không, có liên quan trực tiếp đến câu hỏi hay không.
        - Đưa ra câu trả lời rõ ràng và ngắn gọn, chỉ ra rằng bạn có thể trả lời câu hỏi mà không cần thêm thông tin nào khác.
        - Chú ý một số từ hay được nói tắt: "gói" thay cho "gói cước", "mạng" thay cho "mạng xã hội", "mạng Internet", "dữ liệu" hoặc "nhà mạng Viettel"...
        Định dạng truy vấn đầu vào:
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
        - Lịch sử chat: {self.chat_history}
        - Kiến thức: {local_content}
        - Câu hỏi: {query}
        """

        response = self.llm_for_checking_rag_knowledge.call(prompt)
        log_info(f"check_local_knowledge: response: {response}")
        # return response.strip().lower() == "- trả lời: có"
        return "có" in response.strip().lower()

    def write_local_knowledge_reasoning_query(self, query: str) -> str | None:
        prompt = f"""
        Bạn là một trợ lý ảo thông minh của nhà mạng Viettel, có khả năng giải đáp thắc mắc của người dùng.
        Nhiệm vụ: Xác định xem bạn có thể trả lời câu hỏi của người dùng mà chỉ dựa theo kiến thức đã cho hay không, bằng cách truy vấn từ cơ sở dữ liệu.
        Bạn được cung cấp một cơ sở dữ liệu các gói cước (gọi tắt là gói) của Viettel. Đây là cơ sở dữ liệu dạng bảng, mỗi hàng chứa thông tin của một gói, mỗi cột chứa thuộc tính cụ thể của gói đó.
        Một số hàng có thể trống (optional).
        Các cột của bảng bao gồm:

        - Mã dịch vụ: Thường chính là mã gói cước, có thể là một chuỗi ký tự hoặc số.
        - Thời gian thanh toán: Có thể là Trả trước hoặc Trả sau.
        - Các dịch vụ tiên quyết: Các dịch vụ mà người dùng cần phải đăng ký trước khi có thể sử dụng gói cước này. Được biểu hiện dưới dạng một danh sách các mã dịch vụ, cách nhau bởi dấu phẩy.
        - Giá (VNĐ): Giá của gói cước, được biểu hiện bằng số tiền cụ thể.
        - Chu kỳ (ngày): Thời gian sử dụng của gói cước, được biểu hiện bằng số ngày cụ thể. Sau khi hết thời gian này, gói cước sẽ tự động gia hạn hoặc ngừng hoạt động, xem thông tin ở cột Tự động gia hạn nhé.
        - 4G tốc độ tiêu chuẩn/ngày: Số lượng dữ liệu (dung lượng) 4G tốc độ tiêu chuẩn mà người dùng nhận được mỗi ngày, được biểu hiện bằng số GB. Bạn xem Chú ý 1 bên dưới.
        - 4G tốc độ cao/ngày: Số lượng dữ liệu (dung lượng) 4G tốc độ cao mà người dùng nhận được mỗi ngày, được biểu hiện bằng số GB. Bạn xem Chú ý 1 bên dưới.
        - 4G tốc độ tiêu chuẩn/chu kỳ: Số lượng dữ liệu (dung lượng) 4G tốc độ tiêu chuẩn mà người dùng nhận được trong toàn bộ chu kỳ, được biểu hiện bằng số GB. Bạn xem Chú ý 1 bên dưới.
        - 4G tốc độ cao/chu kỳ: Số lượng dữ liệu (dung lượng) 4G tốc độ cao mà người dùng nhận được trong toàn bộ chu kỳ, được biểu hiện bằng số GB. Bạn xem Chú ý 1 bên dưới.
        - Gọi nội mạng: Số phút gọi miễn phí cho các cuộc gọi nội mạng, cước phí gọi...
        - Gọi ngoại mạng: Số phút gọi miễn phí cho các cuộc gọi ngoại mạng, cước phí gọi...
        - Tin nhắn: Thông tin cước phí tin nhắn SMS, số tin nhắn miễn phí...
        - Chi tiết: Thông tin thêm về gói cước, thường là các điều khoản và điều kiện sử dụng, các dịch vụ miễn phí đi kèm (như thoải mái, thả ga xem Facebook, TikTok...) và có thể có khẩu hiệu (slogan) nếu gói cước dành cho giới trẻ.
        - Tự động gia hạn: Thông tin về việc gói cước có tự động gia hạn hay không sau mỗi chu kỳ sử dụng (xem cột Chu kỳ phía trên), thường là Có hoặc Không.
        - Cú pháp đăng ký: Cú pháp SMS mà người dùng cần sử dụng để đăng ký gói cước này, thường có dạng <mã dịch vụ> gửi <số điện thoại>. Ví dụ SD70 DK8 gửi 290 nghĩa là để đăng ký gói này, người dùng cần soạn tin "SD70 DK8" và gửi đến số 290.

        Chú ý 1: Nếu dữ liệu theo ngày là số dương thì nghĩa là một ngày người dùng chỉ được dùng tối đa bấy nhiêu dữ liệu mà thôi, sang ngày khác lại được thêm. Còn nếu không có dữ liệu theo ngày thì nghĩa là người dùng được dùng thoải mái toàn bộ dữ liệu trong chu kỳ mà không bị giới hạn theo ngày, cho đến khi hết dữ liệu trong chu kỳ đó thì phải chờ chu kỳ tiếp theo (nếu gia hạn) mới được tiếp tục sử dụng.
        Chú ý 1b: Nếu người dùng hỏi dung lượng thì cần chọn các cột sau: "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ", "Chi tiết".
        Chú ý 2: Bạn phải luôn SELECT các cột sau trong mọi trường hợp: "Mã dịch vụ", "Cú pháp đăng ký", "Giá (VNĐ)" và "Chi tiết".
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
        Nếu người dùng hỏi "nhiều data", "data không giới hạn"... thì nên chọn các gói có "4G tốc độ tiêu chuẩn/ngày" REACHES MIN hoặc "4G tốc độ cao/ngày" REACHES MAX, hoặc cột "Chi tiết" CONTAINS "không giới hạn", "thả ga" .v.v.

        Trong trường hợp bạn có thể trả lời câu hỏi của người dùng bằng cách tạo một truy vấn dữ liệu như trên, hãy trả về truy vấn. Nếu không, trả về IMPOSSIBLE.

        Hãy nghiên cứu các ví dụ dưới đây, và trả lời câu hỏi được đưa ra ở cuối cùng:
        Ví dụ 1:
        - Lịch sử chat: Không có
        - Câu hỏi: Gói cước nào có giá rẻ nhất?
        - Trả lời: SELECT "Mã dịch vụ", "Giá (VNĐ)" WHERE "Giá (VNĐ)" REACHES MIN
        Ví dụ 2:
        - Lịch sử chat: Không có
        - Câu hỏi: Làm thế nào để đăng ký dịch vụ SD70?
        - Trả lời: SELECT "Chi tiết", "Cú pháp đăng ký" và "Mã dịch vụ" WHERE "Mã dịch vụ" = "SD70"
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
        - Trả lời: SELECT "Mã dịch vụ", "Cú pháp đăng ký", "Giá (VNĐ)", "Chi tiết" WHERE "Mã dịch vụ" = "SD70"

        Lịch sử chat: {self.chat_history}
        Câu hỏi: {query}
        """
        response = self.llm_for_checking_rag_knowledge.call(prompt)
        return None if "impossible" in response.strip().lower() else response.strip()

    def check_local_knowledge_reasoning(self, query: str):
        q = self.write_local_knowledge_reasoning_query(query)
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

    @classmethod
    def setup_embeddings(cls):
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING,
            model_kwargs={"device": "cuda"},
        )

    @classmethod
    def setup_vector_store(cls, embeddings):
        """Setup vector store from CSV file"""
        log_info("Setting up vector store...")

        log_info("Setting up vector store: Loading documents...")
        df = pd.read_csv("viettel.csv")  # type: ignore
        documents = cls.convert_df_to_documents(df)

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

    @classmethod
    def get_local_content(cls, vector_store, query):
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
        context = cls.combine_documents(docs)

        log_info(
            f"Getting the most relevant local content: Combined context: {context}"
        )
        return context

    @classmethod
    def combine_documents(cls, docs):
        return "\n---\n".join([doc.page_content for doc in docs])

    @staticmethod
    def convert_df_to_documents(df: pd.DataFrame):
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

    def get_answer_from_local_knowledge(self, vector_store, query):
        """Get answer from local knowledge"""
        context = self.get_local_content(vector_store, query)
        if self.check_local_knowledge(query, context):
            return context
        else:
            return None

    def process_query(self, query: str):
        log_info(f"PROCESSING QUERY: {query}")

        context = None

        # Check reasoning-based RAG
        reasoning_retrieval_start_time = pd.Timestamp.now()
        reasoning_data_context = self.check_local_knowledge_reasoning(query)
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
            can_answer_locally = self.check_local_knowledge(query, local_content)
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
            self.generate_final_answer(context, query),
            retrieval_time,
        )

    def generate_final_answer(self, context: str, query: str):
        """Generate final answer using LLM"""
        prompt = f"""Bạn là một trợ lý ảo thông minh, là nhân viên chăm sóc khách hàng của Viettel. Bạn có khả năng trả lời câu hỏi của người dùng dựa trên ngữ cảnh đã cho.
            Bạn có thể sử dụng ngữ cảnh đã cho để trả lời câu hỏi của người dùng một cách chính xác và nhanh chóng.
            Hãy đảm bảo rằng câu trả lời của bạn là chính xác và đầy đủ, dựa trên ngữ cảnh đã cho, cũng như lịch sử trò chuyện. Bạn cần xử lý linh hoạt đặc biệt là các từ đồng nghĩa.
            Lưu ý không được sử dụng Markdown, chẳng hạn không được in đậm chữ như kiểu **này**.
            Bạn cũng cần thêm những từ ngữ hòa nhã, lịch sự, thân thiện, thể hiện sự chuyên nghiệp trong câu trả lời của mình, chẳng hạn "nhé", "Xin chào quý khách", "Dạ, bên em đã nhận được câu hỏi của anh/chị"...
            Chú ý rằng ngữ cảnh là do hệ thống tự sinh ra dựa trên chính câu nói của người dùng, và có thể không chính xác hoàn toàn.
            Hãy tưởng tượng ngữ cảnh là do chính bản thân bạn tìm kiếm, tra cứu mà có. Vì vậy không được nói "hiện tại hệ thống đang tìm kiếm gói cước A, B" mà nên nói "Dạ, em vừa tìm được gói cước A, B có thể đáp ứng nhu cầu của anh/chị ạ"...
            Hãy trả lời người dùng một cách tự nhiên, như thể bạn đang nói chuyện với họ.
            Khi tư vấn gói dữ liệu cần linh hoạt trong ước lượng lượng dữ liệu phù hợp nhu cầu người dùng, tránh cứng nhắc, không cố tìm con số chính xác. Tìm con số gần nhất và tư vấn cho khách gói cước phù hợp nhất.
            Trong mọi trường hợp, bạn KHÔNG được hướng dẫn người dùng liên hệ tổng đài chăm sóc khách hàng của Viettel, hoặc truy cập website của Viettel.

            Dưới đây là tóm tắt lịch sử trò chuyện giữa bạn (Trợ lý ảo) và người dùng (Câu nói của người dùng):
            {self.chat_history}

            Còn đây là câu hỏi của người dùng:
            {query}

            Chú ý nếu người dùng nói một gói nào đó "ổn", "tạm được", "được", "ok", "được rồi" thì bạn cần hiểu là người dùng đã đồng ý với gói cước đó, và bạn cần tư vấn thêm về cách đăng ký, chi tiết, giá tiền... gói cước đó, dựa theo ngữ cảnh được cung cấp.
            Ngữ cảnh như sau:
            {context}
        """,
        # TODO: Bạn cũng chú ý giới tính của người dùng.

        return self.llm_for_checking_rag_knowledge.call_tokenstream(prompt)

    def squeeze_history(self, update: HistoryUpdate):
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
            with self._chat_history_lock:
                prompt = f"""
                Bạn là một thư ký có khả năng tóm tắt lịch sử trò chuyện giữa người dùng và nhân viên chăm sóc khách hàng của Viettel.
                Bạn có thể tóm tắt lịch sử trò chuyện một cách ngắn gọn và súc tích, chỉ giữ lại những thông tin quan trọng nhất.
                Đặc biệt, bạn phải chỉ ra được người dùng đang nói tới gói cước nào (nếu có), và những thông tin quan trọng liên quan đến gói cước đó (nếu có).
                Ví dụ người dùng bảo gói này ổn, thì phải xem trước đó họ nói về gói nào, chỉ ra cụ thể tên gói đó.
                Không được nói "gói đã nêu", "gói mà người dùng thấy ổn"... mà phải nói rõ ví dụ gói SD70.
                Lịch sử trò chuyện trước đây được tóm tắt như sau:
                {self._chat_history}

                {updates_text}

                Bạn hãy tiếp tục tóm tắt lịch sử trò chuyện này một cách ngắn gọn và súc tích, chỉ giữ lại những thông tin quan trọng nhất.
                Hãy chú ý rằng bạn không được phép thay đổi nội dung của lịch sử trò chuyện trước đó, mà chỉ được phép tóm tắt lại nó.
                Bạn cũng không được phép thêm bất kỳ thông tin nào khác ngoài những gì đã có trong lịch sử trò chuyện trước đó.
                Định dạng câu trả lời của bạn: Toàn bộ lịch sử trò chuyện, không dùng Markdown, không dùng in đậm chữ như kiểu **này**.
                """
                self._chat_history = self.llm_for_summarizing.call(prompt)
                log_info(f"squeeze_history(): history updated: {self._chat_history}")
        finally:
            squeeze_history_time_end = pd.Timestamp.now()
            log_info_and_print(f"*** Squeezed history in {(squeeze_history_time_end - squeeze_history_time_start).total_seconds()}.")

    def squeeze_history_in_background(self, update: HistoryUpdate):
        thread = Thread(
            target=self.squeeze_history,
            args=(update,),
        )
        thread.start()

    def main(self):
        for query in self.generate_queries():
            time_start = pd.Timestamp.now()
            answer_iter, rag_time = self.process_query(query)
            time_end = pd.Timestamp.now()

            print(f"Answer: ", end="")
            full_answer = ""
            for answer in answer_iter:
                if answer is None:
                    break
                full_answer += answer
                print(answer, end="", flush=True)
            self.squeeze_history_in_background(HistoryUpdate(assistant=full_answer, user=query, user_first=True))
            print("\n")
            log_info_and_print(f"*** Took {(time_end - time_start).total_seconds()} seconds of which RAG took {rag_time.total_seconds()} seconds.\n")

    def generate_queries(self):
        for query in [
            # "Gói cước nào có giá rẻ nhất?",
            # "Làm thế nào để đăng ký dịch vụ SD70?",
            # "Em ơi thế sao thuê bao của anh cứ tự trừ tiền thế nhỉ, em xem giúp anh số dư còn bao nhiêu với",

            "Ừ thế xem giúp anh gói nào để anh lướt mạng thả ga đi, một ngày xem phim đã tốn mấy gigabyte rồi",
            # "Ơ thế là bên em không có một gói mạng xã hội nào à?",
            "Ừ thế cụ thể thì những gói này ưu đãi như nào em, nên mua gói nào rẻ mà nhiều data chút nhỉ",
            "Ừ thế tư vấn giúp anh cái gói nào rẻ nhất trong số những cái em đã nói nhé",
            "À được rồi cảm ơn em nhé. Anh vừa nhắn tin rồi đấy, em xem thuê bao của anh đã cập nhật gói cước với số dư chưa",
            # "À rồi ok cảm ơn em nhé. Em xem giúp anh gói nào cho mẹ anh nữa, mẹ anh dùng điện thoại cục gạch mà chắc nghe gọi cũng ít thôi",
            "Ok còn thằng bạn anh cần gói gì mà không giới hạn dữ liệu ấy",
            # "Ừ thế cụ thể dung lượng mấy gói này như nào em",
            # "Ừ vậy cho anh gói nào mà 30 g b một tháng ấy",
            # "À ừ anh thấy gói này được đấy, cơ mà em bảo đăng ký như nào ấy nhỉ, anh chưa nghe rõ"

            # "Chắc mẹ anh cũng chả nhắn tin hay dùng ứng dụng gì đâu, điện thoại cục gạch mà em"
            # "Ừ anh thấy gói này ổn đấy em ạ"

            # "Tôi muốn biết về gói cước 4G của Viettel.",
            # "Có gói cước nào cho sinh viên không?",
            # "Gói cước nào có tốc độ cao nhất?",
            # "Tôi muốn biết về chương trình khuyến mãi hiện tại của Viettel.",
            # "Có gói cước nào không giới hạn dữ liệu không?",
            # "Tôi muốn biết về các dịch vụ đi kèm với gói cước 5G.",
        ]:
            print(f"Query: {query}")
            yield query

        print("# Enter your next queries, or type 'exit' to quit.")
        while True:
            query = input("Query: ")
            if query.lower() == "exit":
                break
            yield query
