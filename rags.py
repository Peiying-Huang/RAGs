from agents import Agents
from retrievers import DocsRetriever,HybridRetriever
from collections import defaultdict
from tqdm import tqdm

class RAG:

    def __init__(self):
        self.agent = Agents()

    def _hybrid_search(self, query: str, k: int = 10, rrf_k: int =15)-> list[str]:
        retriever = HybridRetriever(rrf_k)
        search_res = retriever.hybrid_candidates(query, k)
        return search_res
    
    def _rrf_fuse(self, result_lists: list[list[tuple[str, float]]], RRF_k: int = 30) ->list:
        
        id_lists = [[r_id for r_id, _ in results] for results in result_lists]
        scores = defaultdict(float)
        for id_list in id_lists:
            for rank, r_id in enumerate(id_list, start=1):
                scores[r_id] += 1.0 / (RRF_k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    
    def SimpleRAG(self, query: str, k: int = 10, rrf_k: int =15) ->tuple[list, str]:
        article_tuples = self._hybrid_search(query,k, rrf_k)
        article_ids = [int(art_id) for art_id, _ in article_tuples]
        articles = DocsRetriever(article_tuples).documents()
        answer = self.agent.generate(query, articles)
        return article_ids, answer.answer
        
    def ParaRAG(self, query: str, k: int = 10, rrf_k: int =15, RRF_k:int = 30) ->tuple[list, str]:
        para_queries = self.agent.paraphrase(query).paraphrased_queries
        result_lists = [self._hybrid_search(para_q, k, rrf_k) for para_q in para_queries]
        article_tuples = self._rrf_fuse(result_lists, RRF_k)[:k]
        article_ids = [int(art_id) for art_id, _ in article_tuples]

        articles = DocsRetriever(article_tuples).documents()
        answer = self.agent.generate(query, articles)
        return article_ids, answer.answer
    
    def DecomRAG(self, query: str, k: int = 10, rrf_k: int =15, RRF_k:int = 30) ->tuple[list, str]:
        decom_queries = self.agent.decompose(query).decomposed_queries
        result_lists = [self._hybrid_search(decom_q, k, rrf_k) for decom_q in decom_queries]
        article_tuples = self._rrf_fuse(result_lists, RRF_k)[:k]
        article_ids = [int(art_id) for art_id, _ in article_tuples]

        articles = DocsRetriever(article_tuples).documents()
        answer = self.agent.generate(query, articles)
        return article_ids, answer.answer

    def StepbRAG(self, query: str, k: int = 10, rrf_k: int =15, RRF_k:int = 30) ->tuple[list, str]:
        stepb_queries = self.agent.stepback(query).stepback_queries
        result_lists = [self._hybrid_search(stepb_q, k, rrf_k) for stepb_q in stepb_queries]
        article_tuples = self._rrf_fuse(result_lists, RRF_k)[:k]
        article_ids = [int(art_id) for art_id, _ in article_tuples]

        articles = DocsRetriever(article_tuples).documents()
        answer = self.agent.generate(query, articles)
        return article_ids, answer.answer

    def MasterRAG(self, query: str, k: int = 10, rrf_k: int =15, RRF_k:int = 30) ->tuple[list, str]:
        #LLM router
        QueryAgent = self.agent.master(query).agent
        if QueryAgent.lower() == "paraphrase":
            queries = self.agent.paraphrase(query).paraphrased_queries
        elif QueryAgent.lower() == "decompose":
            queries = self.agent.decompose(query).decomposed_queries
        else:
            queries = self.agent.stepback(query).stepback_queries
            
    
        result_lists = [self._hybrid_search(q, k, rrf_k) for q in queries]
        article_tuples = self._rrf_fuse(result_lists, RRF_k)[:k]
        article_ids = [int(art_id) for art_id, _ in article_tuples]

        articles = DocsRetriever(article_tuples).documents()
        answer = self.agent.generate(query, articles)
        return article_ids, answer.answer


    def EvaluateRAG(self, query: str, k: int = 10, rrf_k: int = 15, RRF_k: int = 30, num_queries:int = 3) -> tuple[list, str]:
        ##fuse the search results from queries of each agent
        para_queries = self.agent.paraphrase(query, num_queries).paraphrased_queries
        result_para = [self._hybrid_search(para_q, k, rrf_k) for para_q in para_queries]
        articles_para = self._rrf_fuse(result_para, RRF_k)[:k]

        stepb_queries = self.agent.stepback(query, num_queries).stepback_queries
        result_stepb = [self._hybrid_search(stepb_q, k, rrf_k) for stepb_q in stepb_queries]
        articles_stepb = self._rrf_fuse(result_stepb, RRF_k)[:k]

        decom_queries = self.agent.decompose(query, num_queries).decomposed_queries
        result_decom = [self._hybrid_search(decom_q, k, rrf_k) for decom_q in decom_queries]
        articles_decom = self._rrf_fuse(result_decom, RRF_k)[:k]

        #find the text of articles
        result_list = sum([articles_para, articles_stepb, articles_decom], [])
        article_ids = {int(art_id) for art_id, _ in result_list}
        
        article_tuples = [(id_val, 0) for id_val in article_ids]
        articles = DocsRetriever(article_tuples).documents()
                        
        article_dicts = [{"article_id": a_id, "article": article}
                         for a_id, article in zip(article_ids, articles)]

        #evaluator agents
        high_ids = []
        medium_ids = []
        low_ids =[]
        for art_dict in tqdm(article_dicts, desc="Evaluating articles"):
            evaluation = self.agent.evaluate(query, art_dict["article"])

            article_id = art_dict["article_id"]
            relevance = evaluation.relevance.lower()
           
            if article_id in article_ids:
                if relevance == "high":
                    high_ids.append(article_id)
                elif relevance == "medium":
                    medium_ids.append(article_id)
                elif relevance == "low":
                    low_ids.append(article_id)

        #get the text of top_k article
        topk_ids = (high_ids + medium_ids+low_ids)[:k]
        art_tuples = [(id_val, 0) for id_val in topk_ids]
        selected_articles = DocsRetriever(art_tuples).documents()

        answer = self.agent.generate(query, selected_articles)
        return topk_ids, answer.answer
    


    def JudgeRAG(self, query: str, k: int = 10,
                 rrf_k: int = 15, max_iterations: int = 3, num_queries: int =1) -> tuple[list, str, int]:
        
        current_query = query
        article_tuples = self._hybrid_search(current_query, k, rrf_k)
        article_ids = [int(art_id) for art_id, _ in article_tuples]
        retrieved_arts = DocsRetriever(article_tuples).documents()

        
        for i in range(1,max_iterations+1):
            verdict = self.agent.judge(current_query, retrieved_arts, i)
            
            if verdict.sufficiency.lower() == "sufficient":
                break

            if verdict.strategy.lower()  == "paraphrase":
                current_query  = self.agent.paraphrase(current_query, num_queries).paraphrased_queries[0]
            elif verdict.strategy.lower()   == "decompose":
                current_query = self.agent.decompose(current_query,num_queries).decomposed_queries[0]
            elif verdict.strategy.lower()   == "stepback":
                current_query = self.agent.stepback(current_query,num_queries).stepback_queries[0]

            article_tuples = self._hybrid_search(current_query, k, rrf_k)
            article_ids = [int(art_id) for art_id, _ in article_tuples]
            retrieved_arts = DocsRetriever(article_tuples).documents()      
            
        answer = self.agent.generate(query, retrieved_arts)
        return article_ids, answer.answer
        


"""
def JudgeRAG(self, query: str, k: int = 10,
                 rrf_k: int = 15, max_iterations: int = 5) -> tuple[list, str, int]:
        current_query = query
        seen_ids = set()
        all_documents = []
        iteration = 1
        summary = ""

        pbar = tqdm(range(1, max_iterations + 1), desc="JudgeRAG", unit="iter")
        for iteration in pbar:
            article_tuples = self._hybrid_search(current_query, k, rrf_k)
            article_ids = [int(art_id) for art_id, _ in article_tuples]
            new_ids = [d for d in article_ids if d not in seen_ids]
            new_tuples = [(art_id, 0) for art_id in new_ids]
            seen_ids.update(article_ids)
            
            all_documents.extend(DocsRetriever(new_tuples).documents())
            verdict = self.agent.judge(query, all_documents, summary, iteration)

            if verdict.sufficiency == "sufficient":
                pbar.set_postfix(status="sufficient")
                break
            current_query = verdict.new_search_query
            summary = verdict.summary

        pbar.close()
        answer = self.agent.generate(query, all_documents[:rrf_k])
        return list(seen_ids), answer.answer
"""


"""    def EvaluateRAG(self, query: str, k: int = 10,
                    rrf_k: int =15, RRF_k: int =30) ->tuple[list, str]:
        result_list = []

        ##fuse the search results from queries of each agent
        para_queries = self.agent.paraphrase(query).paraphrased_queries
        result_para = [self._hybrid_search(para_q, k, rrf_k) for para_q in para_queries]
        articles_para = self._rrf_fuse(result_para, RRF_k)[:k]
        
        stepb_queries = self.agent.stepback(query).stepback_queries
        result_stepb = [self._hybrid_search(stepb_q, k, rrf_k) for stepb_q in stepb_queries]
        articles_stepb = self._rrf_fuse(result_stepb, RRF_k)[:k]

        decom_queries = self.agent.decompose(query).decomposed_queries
        result_decom = [self._hybrid_search(decom_q, k, rrf_k) for decom_q in decom_queries]
        articles_decom = self._rrf_fuse(result_decom, RRF_k)[:k]

        #find the text of articles       
        result_list = sum([articles_para, articles_stepb, articles_decom], [])
        article_ids = {int(art_id) for art_id, _ in result_list}

        article_tuples = [(id_val, 0) for id_val in article_ids]
        articles = DocsRetriever(article_tuples).documents()

        article_dicts = [{"article_id": a_id, "article": article}
                         for a_id, article in zip(article_ids, articles)]
        
        #evaluator agents
        details = []
        highart_ids = []
        mediumart_ids = []

        for art_dict in article_dicts:
            evaluation = self.agent.evaluate(query, art_dict)
            
            article_id = evaluation.article_id
            relevance = evaluation.relevance.lower()

            details.append({
                "article_id": article_id,
                "relevance": relevance,
                "justification": evaluation.justification
            })

            if article_id in article_ids:
                if relevance == "high":
                    highart_ids.append(article_id)
                elif relevance == "medium":
                    mediumart_ids.append(article_id)

        #get the text of top_k article
        topk_ids = (highart_ids + mediumart_ids)[:k]
        art_tuples = [(id_val, 0) for id_val in topk_ids]
        selected_articles = DocsRetriever(art_tuples).documents()
        
        answer = self.agent.generate(query, selected_articles)
        return topk_ids, answer.answer
"""






    
