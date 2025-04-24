# from . import HUGGINGFACE_EMBEDDING, HuggingFaceEmbeddings
# from .reasoning_data_query_engine import ReasoningDataQueryEngine
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# import numpy as np

from . import MyRAG

if __name__ == "__main__":
    MyRAG().main()
    # embeddings = HuggingFaceEmbeddings(
    #         model_name=HUGGINGFACE_EMBEDDING,
    #         model_kwargs={"device": "cuda"},
    #     )
    # e = ReasoningDataQueryEngine("viettel.csv", lambda x: np.array(embeddings.embed_query(x)))
    # q = 'SELECT "Mã dịch vụ", "4G tốc độ tiêu chuẩn/ngày", "4G tốc độ cao/ngày", "4G tốc độ tiêu chuẩn/chu kỳ", "4G tốc độ cao/chu kỳ" WHERE "Slogan" CONTAINS "lướt mạng thả ga" AND "Dịch vụ miễn phí" CONTAINS "thả ga" AND "Chu kỳ (ngày)" REACHES MAX AND "4G tốc độ cao/chu kỳ" >= "30"'
    # rd_query = ReasoningDataQueryEngine.compile(q)
    # print(e.apply(rd_query))
    # MyRAG().main()
