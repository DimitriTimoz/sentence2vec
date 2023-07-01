#[cfg(feature = "loading")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(feature = "loading")]
use std::path::Path;

/// Word vector of dimension D.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "loading", derive(Serialize, Deserialize))]
pub struct WordVec<const D: usize> {
    vec: Vec<f32>,
}

impl<const D: usize> WordVec<D> {
    /// Create a new WordVec from a vector of dimension D.
    pub fn new(vec: [f32; D]) -> Self {
        Self { vec: vec.to_vec() }
    }

    /// Get the vector as a slice.
    pub fn get_vec(&self) -> &[f32; D] {
        self.vec.as_slice().try_into().unwrap()
    }

    /// Calculate the cosine similarity of two vectors.
    pub fn cosine(&self, vec: &Self) -> f32 {
        let mut dot = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for (v1, v2) in self.vec.iter().zip(vec.vec.iter()) {
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            eprintln!("Warning: the norm of a vector is 0.");
        }

        dot.powi(2) / (norm1 * norm2)
    }
}

pub trait Word2VecTrait<const D: usize> {
    fn get_vec(&self, word: &str) -> Option<&WordVec<D>>;
}

/// Word2Vec model.
/// Contains a map from words to vectors.
/// D is the dimension of the vectors.
#[cfg_attr(feature = "loading", derive(Serialize, Deserialize))]
pub struct Word2Vec<const D: usize> {
    word_vecs: HashMap<String, WordVec<D>>,
}

impl<const D: usize> Word2Vec<D> {
    /// Load the word2vec model from a text file with the format:
    /// word1 0.1 0.2 0.3 ... 0.1
    /// word1 is the word, and the rest are the vector of dimension D.
    #[cfg(feature = "loading")]
    pub async fn load_from_txt<P>(path: P) -> Option<Self>
    where
        P: AsRef<Path>,
    {
        let mut word_vecs = HashMap::new();
        let lines = crate::file::read_lines(path);
        if let Ok(lines) = lines {
            for line in lines.skip(1) {
                let line = line.unwrap();
                let mut iter = line.split_whitespace();
                let word = iter.next().unwrap();
                let vec = iter.map(|s| s.parse::<f32>().unwrap()).collect::<Vec<_>>();
                word_vecs.insert(word.to_string(), WordVec::new(vec.try_into().unwrap()));
            }
            Some(Self { word_vecs })
        } else {
            None
        }
        
    }

    /// Create a new Word2Vec from a map of words to vectors.
    pub async fn from_word_vecs(word_vecs: HashMap<String, WordVec<D>>) -> Self {
        Self { word_vecs }
    }

    /// Save the word2vec model to a binary file with custom serialization.
    #[cfg(feature = "loading")]
    pub async fn save_to_bytes<P>(&self, path: P)
    where
        P: AsRef<Path>,
    {
        let mut bytes = Vec::new();
        bincode::serialize_into(&mut bytes, &self).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// Load the word2vec model from a binary file with custom serialization.
    #[cfg(feature = "loading")]
    pub async fn load_from_bytes<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let bytes = std::fs::read(path).unwrap();
        bincode::deserialize(&bytes).unwrap()
    }

    /// Calculate the cosine similarity of two words.
    pub fn cosine(&self, word1: &str, word2: &str) -> f32 {
        let vec1 = self.get_vec(word1).unwrap();
        let vec2 = self.get_vec(word2).unwrap();

        vec1.cosine(vec2)
    }

    #[cfg(feature = "partition")]
    /// Partition the word2vec model into f folders for a total of n files.
    /// The words are sorted alphabetically and distributed evenly.
    /// Files and folders are named as the first word they contain.
    pub async fn partition<P>(&self, dist: P, n: usize, f: usize)
    where
        P: AsRef<Path>,
    {
        use log::{info, trace};

        info!("Partitioning into {} folders and {} files", f, n);
        let mut dist = dist.as_ref().to_path_buf();
        dist.push("word2vec");
        std::fs::create_dir_all(&dist).unwrap();

        // Sort the words alphabetically.
        let mut words = self.word_vecs.keys().collect::<Vec<_>>();

        info!("Sorting {} words", words.len());
        words.sort();
        info!("Done sorting");
        // Calculate the number of words per file.
        let words_per_file = words.len() / n;
        let words_per_folder = words.len() / f;

        // Create the folders.
        let mut current_map: HashMap<String, WordVec<D>> = HashMap::new();
        let mut current_folder = dist.clone();
        for (i, word) in words.iter().enumerate() {
            if i % words_per_folder == 0 {
                current_folder = dist.clone();
                current_folder.push(words[i]);
                std::fs::create_dir_all(&current_folder).unwrap();
                trace!("Created folder {}", current_folder.display());
            }

            current_map.insert(word.to_string(), self.get_vec(word).unwrap().clone());

            if i % words_per_file == 0 || i == words.len() - 1 {
                let mut file = current_folder.clone();
                file.push(words[i]);
                file.set_extension("bin");
                let mut bytes = Vec::new();
                bincode::serialize_into(&mut bytes, &current_map).unwrap();
                std::fs::write(file.clone(), bytes).unwrap();
                current_map.clear();
                trace!("Created file {}", file.display());
            }
        }
    }

    /// Get the subset of words that are in the model from the list.
    pub async fn get_subset(&self, words: &[String]) -> Word2Vec<D> {
        let mut word_vecs = HashMap::new();
        for word in words {
            if let Some(vec) = self.get_vec(word) {
                word_vecs.insert(word.to_string(), vec.clone());
            }
        }
        Self { word_vecs }
    }

    /// Get the subset of words that are in the model from wordlist.txt.
    #[cfg(feature = "loading")]
    pub async fn get_subset_from_wordlist<P>(&self, path: P) -> Word2Vec<D>
    where
        P: AsRef<Path>,
    {
        use crate::file::read_lines;

        let mut word_vecs = HashMap::new();
        for word in crate::file::read_lines(path).unwrap().flatten() {
            if let Some(vec) = self.get_vec(&word) {
                word_vecs.insert(word.to_string(), vec.clone());
            }
        }
        Self { word_vecs }
    }
}

impl<const D: usize> Word2VecTrait<D> for Word2Vec<D> {
    fn get_vec(&self, word: &str) -> Option<&WordVec<D>> {
        self.word_vecs.get(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wordvec() {
        let vec1 = WordVec::new([1.0, 0.0]);
        let vec2 = WordVec::new([1.0, 0.0]);
        let vec3 = WordVec::new([0.0, 1.0]);

        // Get the vector as a slice.
        assert_eq!(vec1.get_vec(), &[1.0, 0.0]);

        // Calculate the cosine similarity of two vectors.
        assert_eq!(vec1.cosine(&vec2), 1.0);
        assert_eq!(vec1.cosine(&vec3), 0.0);

        // Test with norm 0.
        let vec1 = WordVec::new([0.0, 0.0]);
        assert!(vec1.cosine(&vec2).is_nan());

    }

    #[tokio::test]
    async fn test_word2vec() {
        let mut word_vecs = HashMap::new();
        word_vecs.insert("word1".to_string(), WordVec::new([1.0, 0.0]));
        word_vecs.insert("word2".to_string(), WordVec::new([0.0, 1.0]));
        let word2vec = Word2Vec::from_word_vecs(word_vecs).await;

        assert_eq!(word2vec.cosine("word1", "word2"), 0.0);
        assert_eq!(word2vec.cosine("word1", "word1"), 1.0);
    }

    #[tokio::test]
    async fn test_word2vec_subset() {
        let mut word_vecs = HashMap::new();
        word_vecs.insert("word1".to_string(), WordVec::new([1.0, 2.0, 3.0]));
        word_vecs.insert("word2".to_string(), WordVec::new([1.0, 2.0, 4.0]));
        let word2vec = Word2Vec::from_word_vecs(word_vecs).await;

        let subset = word2vec.get_subset(&["word1".to_string(), "word3".to_string()]).await;

        assert_eq!(subset.word_vecs.len(), 1);
        assert_eq!(subset.word_vecs.get("word1").unwrap().get_vec(), &[1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "loading")]
    #[tokio::test]
    async fn test_word2vec_load() {
        let word2vec: Word2Vec<3> = Word2Vec::load_from_txt("tests/word2vec.txt").await.unwrap();

        assert!(Word2Vec::<30>::load_from_txt("tests/word2v9+65ds6d5ec.txt").await.is_none());

        assert_eq!(word2vec.word_vecs.len(), 5);

        assert_eq!(word2vec.cosine("chien", "chat"), 0.0);
    }

}
