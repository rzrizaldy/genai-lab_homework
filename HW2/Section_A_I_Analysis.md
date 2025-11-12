# A.I. Analysis of Vector Distance Metrics

## (i) Describe these distance metrics:

### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It is calculated as the dot product of the vectors divided by the product of their magnitudes (L2 norms). The formula is:

\[
\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \times ||\mathbf{B}||} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
\]

The result ranges from -1 to 1, where:
- **1** indicates identical direction (vectors point in the same direction)
- **0** indicates orthogonality (vectors are perpendicular)
- **-1** indicates opposite direction (vectors point in opposite directions)

Cosine similarity is **scale-invariant**, meaning it only considers the direction of vectors, not their magnitude. This makes it particularly useful for comparing documents or text embeddings where the length of the text doesn't necessarily indicate similarity.

### Euclidean Distance

Euclidean distance (also known as L2 distance) measures the straight-line distance between two points in Euclidean space. It is calculated as the square root of the sum of squared differences between corresponding elements of two vectors. The formula is:

\[
\text{Euclidean Distance} = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2} = ||\mathbf{A} - \mathbf{B}||_2
\]

The result is always a non-negative value, where:
- **0** indicates identical vectors
- **Larger values** indicate greater dissimilarity

Euclidean distance is **scale-sensitive**, meaning it considers both the direction and magnitude of vectors. It measures the actual geometric distance between points in the vector space, making it intuitive for spatial reasoning.

### Dot Product

Dot product (also called inner product or scalar product) is the sum of the products of corresponding elements of two vectors. The formula is:

\[
\text{Dot Product} = \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i B_i
\]

The result can be any real number (positive, negative, or zero), where:
- **Positive values** indicate vectors pointing in similar directions
- **Zero** indicates orthogonality
- **Negative values** indicate vectors pointing in opposite directions

The dot product is **not normalized** and is sensitive to both the magnitude and direction of vectors. When vectors are normalized (unit length), the dot product equals cosine similarity. The magnitude of the dot product increases with both the similarity of direction and the magnitudes of the vectors.

---

## (ii) For each of the metrics defined in (i), describe how the metric is different from the other metrics.

### Cosine Similarity vs. Others

**Cosine Similarity vs. Euclidean Distance:**
- Cosine similarity measures the **angle** between vectors (direction-based), while Euclidean distance measures the **straight-line distance** between points (magnitude and direction-based).
- Cosine similarity is **scale-invariant**—multiplying vectors by a constant doesn't change the similarity score. Euclidean distance is **scale-sensitive**—larger vectors will have larger distances.
- Cosine similarity ranges from -1 to 1, while Euclidean distance ranges from 0 to infinity.
- For text embeddings, cosine similarity is often preferred because document length doesn't affect similarity (e.g., a short summary and a long article about the same topic can have high cosine similarity).

**Cosine Similarity vs. Dot Product:**
- Cosine similarity is the **normalized version** of the dot product—it divides the dot product by the product of vector magnitudes.
- Cosine similarity is **scale-invariant**, while dot product is **scale-sensitive** (larger vectors produce larger dot products).
- Both measure directional similarity, but cosine similarity provides a consistent scale (-1 to 1) regardless of vector magnitude.
- When vectors are normalized to unit length, cosine similarity and dot product are equivalent.

### Euclidean Distance vs. Others

**Euclidean Distance vs. Cosine Similarity:**
- Euclidean distance considers **both magnitude and direction**, while cosine similarity considers **only direction**.
- Euclidean distance is a **distance metric** (larger values = more dissimilar), while cosine similarity is a **similarity metric** (larger values = more similar).
- Euclidean distance is sensitive to vector scaling—if you double all values in a vector, the distance changes proportionally. Cosine similarity remains unchanged.
- Euclidean distance is more intuitive for spatial data (e.g., coordinates, physical measurements), while cosine similarity is better for high-dimensional sparse data (e.g., text embeddings, TF-IDF vectors).

**Euclidean Distance vs. Dot Product:**
- Euclidean distance measures **dissimilarity** (distance between points), while dot product measures **similarity** (alignment of directions).
- Euclidean distance is always **non-negative**, while dot product can be **positive, negative, or zero**.
- Euclidean distance considers the **difference** between vectors, while dot product considers their **product**.
- Euclidean distance is a true **metric** (satisfies triangle inequality), while dot product is not a distance metric.

### Dot Product vs. Others

**Dot Product vs. Cosine Similarity:**
- Dot product is the **unnormalized** version of cosine similarity—it doesn't divide by vector magnitudes.
- Dot product is **scale-sensitive**—the magnitude of the result depends on the magnitudes of both input vectors. Cosine similarity is **scale-invariant**.
- Dot product can produce **arbitrarily large values** for large vectors, while cosine similarity is bounded between -1 and 1.
- For normalized vectors (unit length), dot product equals cosine similarity.

**Dot Product vs. Euclidean Distance:**
- Dot product measures **similarity** (higher values = more similar), while Euclidean distance measures **dissimilarity** (higher values = more different).
- Dot product can be **negative** (indicating opposite directions), while Euclidean distance is always **non-negative**.
- Dot product is **not a distance metric**—it doesn't satisfy the triangle inequality. Euclidean distance is a proper metric.
- Dot product is computationally **simpler** (no square root), but Euclidean distance provides more intuitive geometric interpretation.

---

## (iii) For each of the metrics defined in (i), describe one advantage and one disadvantage of using the metric.

### Cosine Similarity

**Advantage:**
Cosine similarity is **scale-invariant**, making it ideal for comparing vectors where magnitude is not meaningful. This is particularly valuable in natural language processing and information retrieval, where document length shouldn't affect similarity. For example, a short summary and a long article about the same topic will have high cosine similarity, even though their magnitudes (word counts) differ significantly. This property makes cosine similarity the preferred metric for text embeddings, TF-IDF vectors, and other high-dimensional sparse data where the direction (semantic meaning) matters more than magnitude.

**Disadvantage:**
Cosine similarity **ignores magnitude information**, which can be a limitation when vector magnitude is meaningful. For instance, in recommendation systems, a user's preference strength (magnitude) might be important—a user who rates movies 4-5 stars is more engaged than one who rates 1-2 stars, even if their rating patterns (directions) are similar. Additionally, cosine similarity treats all dimensions equally and doesn't account for the importance of different features, which can be problematic when some dimensions are more relevant than others for the task at hand.

### Euclidean Distance

**Advantage:**
Euclidean distance provides an **intuitive geometric interpretation** as the straight-line distance between points in space. This makes it easy to understand and visualize, especially in 2D or 3D spaces. It's also a **true metric** that satisfies the triangle inequality, making it suitable for clustering algorithms (like K-means) and nearest neighbor search. Euclidean distance is particularly effective for dense, low-dimensional data where both magnitude and direction are meaningful, such as physical measurements, coordinates, or feature vectors where the scale of values is consistent and interpretable.

**Disadvantage:**
Euclidean distance is **highly sensitive to the scale of features**, which can cause features with larger numerical ranges to dominate the distance calculation. This requires careful feature scaling or normalization before use. Additionally, in high-dimensional spaces, Euclidean distance suffers from the **"curse of dimensionality"**—as dimensions increase, all points become approximately equidistant, making it less discriminative. For sparse, high-dimensional data like text embeddings, Euclidean distance can be less effective than cosine similarity because it doesn't account for the sparsity and scale-invariance properties of such data.

### Dot Product

**Advantage:**
Dot product is **computationally efficient**—it requires only element-wise multiplication and summation, with no square root or normalization operations. This makes it faster to compute than Euclidean distance or cosine similarity, especially for large-scale applications. Additionally, dot product can capture **both direction and magnitude** information simultaneously, which can be useful when both aspects are relevant. For example, in neural networks, dot products are fundamental operations that can be efficiently computed using optimized matrix multiplication libraries, making them the basis for many machine learning algorithms.

**Disadvantage:**
Dot product is **not scale-invariant** and is sensitive to vector magnitudes, which can make comparisons between vectors of different scales problematic. A large-magnitude vector will produce a larger dot product with any other vector, even if the directional similarity is low. This means dot product values are not directly comparable across different vector pairs unless the vectors are normalized. Additionally, dot product is **not a distance metric**—it doesn't satisfy properties like the triangle inequality, which limits its use in certain algorithms that require proper distance metrics. The unbounded nature of dot product values (can be arbitrarily large) also makes it difficult to interpret similarity scores without normalization.

---

## Summary

In the context of RAG (Retrieval-Augmented Generation) systems like the one implemented in Section A:

- **Cosine Similarity** (via normalized embeddings with IndexFlatIP) is the optimal choice for text retrieval because it focuses on semantic similarity regardless of document length, making it ideal for finding relevant policy sections.

- **Euclidean Distance** would be less effective here because policy documents vary significantly in length, and we care more about semantic content than document size.

- **Dot Product** (when used with normalized embeddings) is equivalent to cosine similarity and provides the computational efficiency needed for real-time retrieval in vector databases.

The implementation uses **FAISS IndexFlatIP** with **normalized embeddings**, which effectively computes cosine similarity through the dot product of normalized vectors—combining the semantic benefits of cosine similarity with the computational efficiency of dot product operations.

