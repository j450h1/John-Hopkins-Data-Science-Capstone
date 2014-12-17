#Load preprocessed ngram dataframes
# df4 <- read.table('df4.txt', fill = TRUE, stringsAsFactors = FALSE) 
# df3 <- read.table('df3.txt', fill = TRUE, stringsAsFactors = FALSE) 
# df2 <- read.table('df2.txt', fill = TRUE, stringsAsFactors = FALSE)
# input <- "my name is"

predict_word <- function(input,df2,df3,df4){
  #Format the input by the user
  input <- tolower(input)
  output <- "" #Will be the predicted word 
  test <- 0   #checking which if/else statement is used
  #Split into 3 words
  wordlist <- unlist(strsplit(input, split = ' '))
  first <- wordlist[1]; second <- wordlist[2]; third <- wordlist[3]  
  #Convert to datatables if necessary
  # library('data.table')
  # dt4 <- data.table(df4); dt3 <- data.table(df3); dt2 <- data.table(df4)
  #rm(c(df4,df3,df2)
  
  #See if the input has an exact match with either n-gram, using a backoff model
  #Start with 4gram down to bigram if necessary
  #If no bigram found, return "the" which is the most common result
  
  library(sqldf)
  fourgram_query <- "SELECT * 
                     FROM df4 
                     WHERE V1 = '$first' AND V2 = '$second' AND V3 = '$third' "
  trigram_query <- "SELECT * 
                    FROM df3 
                    WHERE V1 = '$second' AND V2 = '$third' "
  bigram_query <- "SELECT * 
                   FROM df2 
                   WHERE V1 = '$third' "
  
  if (nrow(fn$sqldf(fourgram_query)) > 0){ #This means all three words found together
    # Find the 4gram with the maximum frequency out of this subset
    result <- fn$sqldf(fourgram_query)
    output <-  result$V4[which.max(result$V5)]
    test <- 4
  } else if (nrow(fn$sqldf(trigram_query)) > 0){ 
    # All three words are not found together so we look at second and third word
    result <- fn$sqldf(trigram_query)  
    output <-  result$V3[which.max(result$V4)]
    test <- 3
  } else if (nrow(fn$sqldf(bigram_query)) > 0){
    # Neither all three words or the preceding two words are found, so we only look for the word prior
    result <- fn$sqldf(bigram_query)  
    output <- result$V2[which.max(result$V3)]
    test <- 2
  } else {
    output <- 'the'
    test <- 1
  #predict the most common word "the" if its not in the model
  }
  return(output)
}  
#predicted = predict_word(input,df2,df3,df4)
#print(predicted)
#Query the dataframe
  #If we want to give choices
  #possible_choices = list()
  #possible_choices = append(value, possible_choices)
  #Split columns into words 
