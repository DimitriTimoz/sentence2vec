use sentence2vec::prelude::*;

#[tokio::test]
async fn test_get_sentence_vector() {
    let word2vec = Word2Vec::<300>::load_from_bytes("tests/common.en.300.bin").await.unwrap();
    let sentence2vec = Sentence2Vec::new(Box::new(word2vec));
    let vec = sentence2vec.get_vec("This is a test.").await.unwrap();
    assert_eq!(vec.get_vec().len(), 300);

    let vec = sentence2vec.get_vec("This is a test.").await.unwrap();
    assert_eq!(vec.get_vec().len(), 300);

    assert!(sentence2vec.get_vec("").await.is_none());
}

#[tokio::test]
async fn test_get_similarity() {
    let word2vec = Word2Vec::<300>::load_from_bytes("tests/common.en.300.bin").await.unwrap();
    let sentence2vec = Sentence2Vec::new(Box::new(word2vec));
    let similarity = sentence2vec.cosine("This is a test.", "This is a test.").await.unwrap();
    assert_eq!(similarity, 1.0);

    let similarity_fruits = sentence2vec.cosine("I love apples.", "I love pears").await.unwrap();

    let similarity_fuits_cars = sentence2vec.cosine("I love apples.", "I love cars").await.unwrap();

    assert!(similarity_fruits > similarity_fuits_cars);
}