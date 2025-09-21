
import sys
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSES:
    """
    Implementation of the Mixture of Stylistic Experts (MoSEs) framework
    for AI-generated text detection based on the research paper.
    """
    
    def __init__(self, n_prototypes=5, pca_components=32, random_state=42):
        """
        Initialize the MoSEs framework.
        
        Args:
            n_prototypes: Number of prototypes for stylistics-aware routing
            pca_components: Number of PCA components for semantic feature compression
            random_state: Random seed for reproducibility
        """
        self.n_prototypes = n_prototypes
        self.pca_components = pca_components
        self.random_state = random_state
        self.prototypes = None
        self.pca = None
        self.scaler = None
        self.cte_model = None
        self.srr_data = None
        
    def extract_linguistic_features(self, texts):
        """
        Extract linguistic features from texts as described in the paper.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of linguistic features
        """
        logger.info("Extracting linguistic features from texts")
        
        features = []
        for text in texts:
            # Basic text features
            text_length = len(text.split())
            
            # Placeholder for log-probability features (would come from LLM in real implementation)
            log_prob_mean = np.random.normal(0, 1)  # Simulated
            log_prob_var = np.random.normal(1, 0.5)  # Simulated
            
            # N-gram repetition features
            words = text.split()
            if len(words) >= 2:
                bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                bigram_rep = len(bigrams) - len(set(bigrams)) / max(1, len(bigrams))
            else:
                bigram_rep = 0
                
            if len(words) >= 3:
                trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
                trigram_rep = len(trigrams) - len(set(trigrams)) / max(1, len(trigrams))
            else:
                trigram_rep = 0
                
            # Type-token ratio
            if len(words) > 0:
                ttr = len(set(words)) / np.sqrt(len(words))
            else:
                ttr = 0
                
            features.append([text_length, log_prob_mean, log_prob_var, 
                           bigram_rep, trigram_rep, ttr])
        
        return np.array(features)
    
    def extract_semantic_embeddings(self, texts):
        """
        Extract semantic embeddings from texts using a simplified approach.
        In a real implementation, this would use BGE-M3 or similar model.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of semantic embeddings
        """
        logger.info("Extracting semantic embeddings from texts")
        
        # Simplified embedding extraction (would use pre-trained model in practice)
        embeddings = []
        for text in texts:
            # Simulate embedding extraction with random vectors
            embedding = np.random.normal(0, 1, 512)  # Simulated 512-dim embedding
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def create_prototypes(self, semantic_embeddings, n_clusters=5):
        """
        Create prototypes using k-means clustering on semantic embeddings.
        
        Args:
            semantic_embeddings: Array of semantic embeddings
            n_clusters: Number of clusters/prototypes to create
            
        Returns:
            Array of prototype vectors
        """
        logger.info(f"Creating {n_clusters} prototypes from semantic embeddings")
        
        # Simplified prototype creation (would use proper clustering in practice)
        # For demonstration, we'll use random selection as prototypes
        np.random.seed(self.random_state)
        indices = np.random.choice(len(semantic_embeddings), n_clusters, replace=False)
        prototypes = semantic_embeddings[indices]
        
        return prototypes
    
    def stylistics_aware_routing(self, input_embedding, prototypes):
        """
        Route input text to appropriate prototypes based on semantic similarity.
        
        Args:
            input_embedding: Semantic embedding of input text
            prototypes: Array of prototype vectors
            
        Returns:
            Indices of nearest prototypes
        """
        # Calculate distances to all prototypes
        distances = cdist([input_embedding], prototypes, metric='cosine')[0]
        
        # Get indices of nearest prototypes
        nearest_indices = np.argsort(distances)[:self.n_prototypes]
        
        logger.info(f"Selected {self.n_prototypes} nearest prototypes")
        return nearest_indices
    
    def prepare_conditional_features(self, linguistic_features, semantic_embeddings):
        """
        Prepare conditional features by combining linguistic features and
        compressed semantic embeddings.
        
        Args:
            linguistic_features: Array of linguistic features
            semantic_embeddings: Array of semantic embeddings
            
        Returns:
            Array of combined conditional features
        """
        logger.info("Preparing conditional features")
        
        # Apply PCA to semantic embeddings
        if self.pca is None:
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            semantic_reduced = self.pca.fit_transform(semantic_embeddings)
        else:
            semantic_reduced = self.pca.transform(semantic_embeddings)
        
        # Combine linguistic and semantic features
        conditional_features = np.hstack([linguistic_features, semantic_reduced])
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            conditional_features = self.scaler.fit_transform(conditional_features)
        else:
            conditional_features = self.scaler.transform(conditional_features)
            
        return conditional_features
    
    def train_cte(self, conditional_features, discrimination_scores, labels):
        """
        Train the Conditional Threshold Estimator (CTE) model.
        
        Args:
            conditional_features: Array of conditional features
            discrimination_scores: Array of discrimination scores
            labels: Array of binary labels (0=AI-generated, 1=human)
            
        Returns:
            Trained CTE model
        """
        logger.info("Training Conditional Threshold Estimator")
        
        try:
            # Combine conditional features with discrimination scores
            X = np.hstack([conditional_features, discrimination_scores.reshape(-1, 1)])
            
            # Train logistic regression model (as described in the paper)
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
            
            model.fit(X, labels)
            logger.info("CTE model training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error training CTE model: {str(e)}")
            sys.exit(1)
    
    def fit(self, texts, discrimination_scores, labels):
        """
        Train the complete MoSEs framework.
        
        Args:
            texts: List of training texts
            discrimination_scores: Array of discrimination scores from base model
            labels: Array of binary labels (0=AI-generated, 1=human)
        """
        logger.info("Starting MoSEs framework training")
        
        try:
            # Extract features
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            
            # Create prototypes for stylistics-aware routing
            self.prototypes = self.create_prototypes(semantic_embeddings)
            
            # Prepare conditional features
            conditional_features = self.prepare_conditional_features(
                linguistic_features, semantic_embeddings
            )
            
            # Store SRR data for reference
            self.srr_data = {
                'conditional_features': conditional_features,
                'discrimination_scores': discrimination_scores,
                'labels': labels,
                'semantic_embeddings': semantic_embeddings
            }
            
            # Train CTE model
            self.cte_model = self.train_cte(conditional_features, discrimination_scores, labels)
            
            logger.info("MoSEs framework training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in MoSEs training: {str(e)}")
            sys.exit(1)
    
    def predict(self, texts, discrimination_scores):
        """
        Predict labels for new texts using the trained MoSEs framework.
        
        Args:
            texts: List of texts to classify
            discrimination_scores: Array of discrimination scores from base model
            
        Returns:
            Array of predicted labels and confidence scores
        """
        logger.info("Making predictions with MoSEs framework")
        
        try:
            # Extract features from input texts
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            
            # Prepare conditional features
            conditional_features = self.prepare_conditional_features(
                linguistic_features, semantic_embeddings
            )
            
            # Combine with discrimination scores
            X = np.hstack([conditional_features, discrimination_scores.reshape(-1, 1)])
            
            # Make predictions
            predictions = self.cte_model.predict(X)
            confidence = self.cte_model.predict_proba(X).max(axis=1)
            
            logger.info("Prediction completed successfully")
            return predictions, confidence
            
        except Exception as e:
            logger.error(f"Error in MoSEs prediction: {str(e)}")
            sys.exit(1)

def simulate_discrimination_scores(texts, is_human=True):
    """
    Simulate discrimination scores from a base model.
    In a real implementation, this would come from RoBERTa, Fast-DetectGPT, etc.
    
    Args:
        texts: List of texts
        is_human: Whether texts are human-written (affects score distribution)
        
    Returns:
        Array of discrimination scores
    """
    if is_human:
        # Human-written texts tend to have higher scores
        return np.random.normal(0.7, 0.2, len(texts))
    else:
        # AI-generated texts tend to have lower scores
        return np.random.normal(0.3, 0.2, len(texts))

def main():
    """
    Main function to demonstrate the MoSEs framework.
    """
    logger.info("Starting MoSEs framework demonstration")
    
    try:
        # Generate sample data (in practice, this would come from real datasets)
        np.random.seed(42)
        
        # Create sample human-written texts
        human_texts = [
            "The quick brown fox jumps over the lazy dog. This is a classic example of English text.",
            "Scientific research requires rigorous methodology and careful analysis of results.",
            "Journalistic writing should be clear, concise, and objective in its presentation of facts.",
            "Literary works often explore complex themes through nuanced character development.",
            "Academic papers follow specific formatting guidelines and citation styles."
        ] * 20  # Repeat to create more samples
        
        # Create sample AI-generated texts
        ai_texts = [
            "The rapid advancement of artificial intelligence has transformed many industries.",
            "Machine learning algorithms can process large datasets to identify patterns.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require substantial computational resources for training.",
            "AI systems can generate human-like text based on patterns in training data."
        ] * 20  # Repeat to create more samples
        
        # Combine texts and create labels
        all_texts = human_texts + ai_texts
        labels = np.array([1] * len(human_texts) + [0] * len(ai_texts))
        
        # Simulate discrimination scores from base model
        human_scores = simulate_discrimination_scores(human_texts, is_human=True)
        ai_scores = simulate_discrimination_scores(ai_texts, is_human=False)
        all_scores = np.concatenate([human_scores, ai_scores])
        
        # Initialize and train MoSEs framework
        meses = MoSES(n_prototypes=3, pca_components=10)
        meses.fit(all_texts, all_scores, labels)
        
        # Create test data
        test_human_texts = [
            "Historical analysis provides valuable insights into contemporary issues.",
            "Creative writing allows authors to express unique perspectives and ideas."
        ]
        test_ai_texts = [
            "Neural networks can be trained on diverse datasets to improve performance.",
            "Transformer architectures have revolutionized natural language processing."
        ]
        test_texts = test_human_texts + test_ai_texts
        test_labels = np.array([1, 1, 0, 0])
        
        # Simulate test discrimination scores
        test_human_scores = simulate_discrimination_scores(test_human_texts, is_human=True)
        test_ai_scores = simulate_discrimination_scores(test_ai_texts, is_human=False)
        test_scores = np.concatenate([test_human_scores, test_ai_scores])
        
        # Make predictions
        predictions, confidence = meses.predict(test_texts, test_scores)
        
        # Evaluate performance
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        # Print results
        logger.info("=" * 50)
        logger.info("MoSEs FRAMEWORK DEMONSTRATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        logger.info("")
        logger.info("Detailed Predictions:")
        for i, (text, true_label, pred_label, conf) in enumerate(zip(
            test_texts, test_labels, predictions, confidence
        )):
            true_type = "Human" if true_label == 1 else "AI-generated"
            pred_type = "Human" if pred_label == 1 else "AI-generated"
            logger.info(f"Text {i+1}:")
            logger.info(f"  True: {true_type}, Predicted: {pred_type} (Confidence: {conf:.4f})")
            logger.info(f"  Sample: {text[:50]}...")
            logger.info("")
        
        logger.info("MoSEs framework demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
