import utils.logging as logger 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils.constants import l1_tags,l2_tags,stopwords

model = SentenceTransformer('all-MiniLM-L6-v2')

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)
    
def precompute_embeddings(tags):
    return {tag: model.encode(tag) for tag in tags}

l1_embeddings = precompute_embeddings(l1_tags.keys())
l2_embeddings = precompute_embeddings(l2_tags.keys())

def classify_event(description):
    newDesc=remove_stopwords(description)
    description_embedding = model.encode(newDesc)
    logger.log_message(message=f"description_embedding {description_embedding}",level="info")
    # Level 1 matching: Find the best match for Level 1 tags
    level_1_tag = None
    level_1_score = 0
    l1_reason = ""
    for keyword, tag_embedding in l1_embeddings.items():
        score = cosine_similarity([description_embedding], [tag_embedding])[0][0]
        if score > level_1_score:
            level_1_score = score
            level_1_tag = l1_tags[keyword]
            l1_reason = keyword
            logger.log_message(message=f"Keyword: {keyword}, Embedding: {tag_embedding}",level="info")
            logger.log_message(message=f"score: {score} | tag: {level_1_tag} | keyword: {keyword}",level="info")
    # Filter L2 tags based on the chosen L1 tag
    filtered_l2_tags = {
        keyword: tags
        for keyword, tags in l2_tags.items()
        if level_1_tag in tags
    }
    # Level 2 matching: Find the best match for Level 2 tags
    level_2_tag = None
    level_2_score = 0
    l2_reason = ""
    for keyword, tags in filtered_l2_tags.items():
        score = cosine_similarity([description_embedding], [l2_embeddings[keyword]])[0][0]
        if score > level_2_score:
            level_2_score = score
            level_2_tag = tags
            l2_reason = keyword

    return {
        "level_1_tag": level_1_tag,
        "level_2_tag": level_2_tag,
        "level_1_reason": l1_reason,
        "level_2_reason": l2_reason
    }

