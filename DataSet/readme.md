**About Problem Statement**:
 - Genre: NLP
 - Problem Type: Contextual Semantic Similarity, Auto generate Text-based answers

**Submission Format**:
 - You need to generate upto 3 distractors for each Question-Answer combination
 - Each distractor is a string
 - The 3 distractors/strings need to be separated with a comma (,)
 - Each value in Results.csv's distractor column will contain the distractors as follows:
	distractor_for_QnA_1 = "distractor1","distractor2","distractor3"

**About the Evaluation Parameter**:
 - All distractor values for 1 question-answer will be converted into a vector form
 - 1 vector gets generated for submitted distractors and 1 vector is generated for truth value
 - cosine_similarity between these 2 vectors is evaluated
 - Similarly, cosine_similarity gets evaluated for all the question-answer combinations
 - Score of your submitted prediction file = mean ( cosine_similarity between distractor vectors for each entry in test.csv)

 
**Common Issues Faced : How to handle them?**:

 - **Download Dataset giving XML error**: Try restarting your session after clearing browser cache/cookies and try again. If you still face an issue, please raise a ticket with us.
 - **Upload Prediction File not working**: Ensure you are compliant with the Guidelines and FAQs. You will face this error if you exceed the maximum number of prediction file uploads allowed.
 - **Exceptions (Incorrect number of Rows / Incorrect Headers / Prediction missing for a key)** : For this problem statement, we recommend you to update the 'distractor' column in Results.csv with your predictions, following the format explained above
 - **Evaluation is getting stuck in loop** : We recommend you to immediately refresh your session and start afresh with a cleared cache. Please ensure your predictions.csv matches the file format Results.csv. Plesae check that all the above mentioned checks have been conducted. If you still face an issue, please raise a ticket with us.	
