use crate::word2vec::{Word2Vec, WordVec, Word2VecTrait};

/// Sentence2Vec model.
/// Contains a Word2Vec model.
pub struct Sentence2Vec<const D: usize> {
    word2vec: Word2Vec<D>,
}

impl<const D: usize> Sentence2Vec<D> {
    pub fn new(word2vec: Word2Vec<D>) -> Self {
        Self { word2vec }
    }

    /// Returns the vector representation of the sentence by averaging the vectors of the words in the sentence. 
    pub fn get_vec(&self, sentence: &str) -> Option<WordVec<D>> {
        let sentence = sentence.to_lowercase();
        // Trim punctuation
        let sentence = sentence.trim_matches(|c: char| !c.is_alphanumeric());
        
        let mut vec = vec![0.0; D];
        let mut count = 0;
        for word in sentence.split_whitespace() {
            let word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(word_vec) = self.word2vec.get_vec(word) {
                for (v, w) in vec.iter_mut().zip(word_vec.get_vec()) {
                    *v += w;
                }
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            for v in vec.iter_mut() {
                *v /= count as f32;
            }
            Some(WordVec::new(vec.try_into().unwrap()))
        }
    }

    pub fn cosine(&self, sentence1: &str, sentence2: &str) -> Option<f32> {
        if let (Some(vec1), Some(vec2)) = (self.get_vec(sentence1), self.get_vec(sentence2)) {
            Some(vec1.cosine(&vec2))
        } else {
            None
        }
    }
}