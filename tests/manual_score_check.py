from src.ingest import load_qa_pairs, build_documents, build_vectorstore

pairs = load_qa_pairs("data/kb/qa_pairs.json")
docs = build_documents(pairs)
vectorstore = build_vectorstore(docs)

queries = [
    "how do I reset my password",
    "where do I park my car",
    "what is the coffee machine schedule",
]

for q in queries:
    print(f"--- {q!r} ---")
    results = vectorstore.similarity_search_with_score(q, k=2)
    for doc, score in results:
        print(f"  score={score:.3f}  id={doc.metadata['id']}  q={doc.metadata['question']}")


#############  Retrieve with scores test  #############

from src.rag_agent import retrieve_with_scores

print("\n--- testing retrieve_with_scores ---")
results = retrieve_with_scores(vectorstore, "how do I reset my password")
for doc, score in results:
    print(f"  score={score:.3f}  id={doc.metadata['id']}")


#############  Is confident match test  #############
from src.rag_agent import is_confident_match

print("\n--- testing is_confident_match ---")
test_queries = [
    "how do I reset my password",       # should be confident
    "where do I park my car",           # should NOT be confident
    "what is the coffee machine schedule",  # should NOT be confident
    "how many sick days do I get",      # should be confident
]
for q in test_queries:
    results = retrieve_with_scores(vectorstore, q)
    confident = is_confident_match(results)
    top_score = results[0][1] if results else None
    print(f"  {q!r}: confident={confident}  top_score={top_score}")       # !r means "repr" — prints the string with quotes around it, so you can see whitespace and punctuation clearly around q. 


############## Stress test: run the same query multiple times to see if scores are consistent ##############
print("\n--- stress testing is_confident_match ---")
stress_queries = [
    "my authenticator app broke, how do I fix it",   # real topic (MFA reset), very different wording — predict: True
    "my monitor keeps flickering",                    # gap topic, shares "monitor"/equipment vocab with onb-003 — predict: False
    "leave",                                           # very short/vague, real general topic — predict: uncertain, worth seeing
    "hey so my vpn thing keeps dropping when im on hotel wifi is that normal", # real topic (VPN/NAT timeout), messy phrasing — predict: True
    "can I bring my dog to the office",                # gap topic, no vocab overlap with anything — predict: False
]
for q in stress_queries:
    results = retrieve_with_scores(vectorstore, q)
    confident = is_confident_match(results)
    top_id = results[0][0].metadata['id'] if results else None
    top_score = results[0][1] if results else None
    print(f"  {q!r}")      # !r means "repr" — prints the string with quotes around it, so you can see whitespace and punctuation clearly around q.
    print(f"    -> confident={confident}  top_score={top_score}  matched_id={top_id}")




############ #  Testing assess_answer - RAG assessment test  #############
from src.rag_agent import assess_answer

print("\n--- testing assess_answer ---")
test_qs = [
    "how do I reset my password",              # should be answerable
    "how many sick days do part-time employees get",  # should NOT be answerable — KB doesn't distinguish part-time
]

for q in test_qs:
    results = retrieve_with_scores(vectorstore, q)
    assessment = assess_answer(q, results)
    print(f"  Q: {q!r}")
    print(f"    can_answer={assessment.can_answer}")
    print(f"    answer={assessment.answer!r}")


############### Test unified entry point end-to-end, answer_question  ###############
from src.rag_agent import answer_question

print("\n--- testing answer_question (full pipeline) ---")
final_tests = [
    "how do I reset my password",                      # confident + answerable
    "where do I park my car",                          # not confident (gap)
    "how many sick days do part-time employees get",   # confident retrieval, but LLM should say can't answer
]
for q in final_tests:
    result = answer_question(vectorstore, q)
    print(f"  Q: {q!r}")
    print(f"    can_answer={result.can_answer}  answer={result.answer!r}")



############# Test Jira ticket creation #############
from src.jira_tool import create_ticket

print("\n--- testing create_ticket ---")
result = create_ticket(
    email="test@boldagent.com",
    summary="Test ticket from SmartDesk AI",
    description="This is a test ticket created during development.",
    category="IT",
    priority="medium"
)
print(result)

############## Test ticket store (SQLite) #############

# add to tests/manual_score_check.py or a new file
from src.ticket_store import link_ticket, get_ticket_keys_for_email

print("\n--- testing ticket_store ---")
link_ticket("test@boldagent.com", "KAN-2")
link_ticket("test@boldagent.com", "KAN-3")
keys = get_ticket_keys_for_email("test@boldagent.com")
print(f"Tickets for test@boldagent.com: {keys}")

no_keys = get_ticket_keys_for_email("nobody@boldagent.com")
print(f"Tickets for nobody@boldagent.com: {no_keys}")


###### Test Jira ticket status retrieval #############
# add to tests/manual_score_check.py
from src.jira_tool import get_ticket_status

print("\n--- testing get_ticket_status ---")
statuses = get_ticket_status("test@boldagent.com")
for s in statuses:
    print(f"  {s}")

no_tickets = get_ticket_status("nobody@boldagent.com")
print(f"  No tickets case: {no_tickets}")


