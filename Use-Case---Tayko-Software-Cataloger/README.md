Applied Data Science for Business 
# Use-Case---Tayko-Software-Cataloger-
Tayko is a software catalog firm that sells games and educational software. It started out as a software manufacturer and later added third-party titles to its offerings. It has recently put together a revised collection of items in a new catalog, which it is preparing to roll out in a mailing. In this use case, I will apply logistic regression, multiple linear regression, and regression trees to make recommendations to optimize the mailing catalog and gross profits. 

# Use Case - Tayko Software Cataloger)
Note: Although Tayko is a hypothetical company, the data in this case (modified slightly for illustrative purposes) were supplied by a real company that sells software through direct sales. The concept of a catalog consortium is based on the Abacus Catalog Alliance.
## BACKGROUND
Tayko is a software catalog firm that sells games and educational software. It started out as a software manufacturer and later added third-party titles to its offerings. It has recently put together a revised collection of items in a new catalog, which it is preparing to roll out in a mailing. 
In addition to its own software titles, Tayko’s customer list is a key asset. In an attempt to expand its customer base, it has recently joined a consortium of catalog firms that specialize in computer and software products. The consortium affords members the opportunity to mail catalogs to names drawn from a pooled list of customers. Members supply their own customer lists to the pool, and can “withdraw” an equivalent number of names each quarter. Members are allowed to do predictive modeling on the records in the pool so they can do a better job of selecting names from the pool.
## THE MAILING EXPERIMENT
Tayko has supplied its customer list of 200,000 names to the pool, which totals over 5,000,000 names, so it is now entitled to draw 200,000 names for a mailing. Tayko would like to select the names that have the best chance of performing well, so it conducts a test—it draws 20,000 names from the pool and does a test mailing of the new catalog.
This mailing yielded 1065 purchasers, a response rate of 0.053. To optimize the performance of the data mining techniques, it was decided to work with a stratified sample that contained equal numbers of purchasers and nonpurchasers. For ease of presentation, the dataset for this case includes just 1000 purchasers and 1000 non purchasers, an apparent response rate of 0.5. Therefore, after using the dataset to predict who will be a purchaser, we must adjust the purchase rate back down by multiplying each case’s “probability of purchase” by 0.053/0.5, or 0.107. 
## DATA
There are two outcome variables in this case. Purchase indicates whether or not a prospect responded to the test mailing and purchased something. Spending indicates, for those who made a purchase, how much they spent. The overall procedure in this case will be to develop two models. One will be used to classify records as purchase or no purchase. The second will be used for those cases that are classified as purchase and will predict the amount they will spend.
### Description of Variables for Tayko Dataset
-US - US address? - Binary - [ 1: Yes ] [ 2: No ]. 
-Source_*  -  Source catalog for the record - Binary - [ 1: Yes ] [ 2: No ].   
(15 possible sources ). 
-Freq - Number of transactions in last year catalog - Numerical.   
-Last_update_days_ago - How many days ago was last update to record - Numerical. 
-1st_update_days_ago - How many days ago first update to customer record - Numerical.   
-Web order -Customer place at least one order via web - Binary - [ 1: Yes ] [ 2: No ].   
-Gender=male - Customer is a male - Binary - [ 1: Yes ] [ 2: No ].   
-Address_is_res - Address is residence - Binary - [ 1: Yes ] [ 2: No ].   
-Purchase -  Person made purchase in test mailing - Binary - [ 1: Yes ] [ 2: No ].    
-Spending - Amount dollars spent by customer in test mailing - Numerical - [ 1: Yes ] [ 2: No ].   
 
 
### The following is a direction and code flow for the task by Tayko:   
1. Each catalog costs approximately $2 to mail (including printing, postage, and mailing costs). Estimate the gross profit that the firm could expect from the remaining 180,000 names if it selects them randomly from the pool.  
2. Develop a model for classifying a customer as a purchaser or nonpurchaser.   
3. Run logistic regression with L2 penalty, using method LogisticRegressionCV, to select the best subset of variables, then use this model to classify the data into purchasers and nonpurchasers. Use only the training set for running the model. (Logistic regression is used because it yields an estimated “probability of purchase,” which is required later in the analysis.    
4. Develop a model for predicting spending among the purchasers.    
5. Develop models for predicting spending with the filtered datasets, using:   
5.1 Multiple linear regression (use stepwise regression).  
5.2 Regression Trees.  
6. Evaluate the models based on performance on the validation data and explain the reasoning for selecting it.    
7. Return to the original test data partition. Note that this test data partition includes both purchasers and nonpurchasers. Create a new data frame called Score Analysis that contains the test data portion of this dataset.    
8. Add a column to the data frame with the predicted scores from the logistic regression.   
9. Add another column with the predicted spending amount from the prediction model chosen.   
10. To adjust for oversampling the purchasers, insert the adjusted probability of purchase      
11. Add a column for expected spending.   
12. Plot the cumulative gains chart of the expected spending (cumulative expected spending as a function of number of records targeted).   
13. Using this cumulative gains curve, estimate the gross profit that would result from mailing to the 180,000 names on the basis of your data mining models.   
14. Briefly explain, in two to three paragraphs, the business objective, the data mining models used, why they were used, the model results, and your recommendations to your non-technical stakeholder team.    
 

