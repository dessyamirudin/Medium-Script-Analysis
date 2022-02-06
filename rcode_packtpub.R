# load library
library(tidyverse)
library(ggplot2)
library(tidytext)
library(topicmodels)
library(textmineR)
library(ldatuning)

# Simple analysis that would be done
# a. Categorize the programming language used
#    Use top words, select top 10 programming words. The rest categorized as others
# b. Categorize the book based on title - Data Science, Software Engineering, Toolings, Other
#    Use topic modeling
# c. How to make catchy title.
#    What is term used mostly for a title. Is there any correlation between words
#    Unfortunately, I haven't scrape the rating data from the web


# loading data
pct = read.csv("packtpub_book.csv")

pct$price = substr(pct$Price,2,length(pct$Price))
pct$price = as.numeric(pct$price)

str(pct)

# categorizing book based on programming language or other
# token book based on the title
# source: https://towardsdatascience.com/beginners-guide-to-lda-topic-modelling-with-r-e57a5a8e7a25

token_title = pct %>% select(Title) %>% mutate(id=row_number()) %>% unnest_tokens(word,Title)

# most used words
top_word = token_title %>% count(word,sort=TRUE)
write.csv(top_word,"top_word.csv")

# removing stop words
# some information miss because of stop words. to create the word count, put stop words later
data("stop_words")
token_title = token_title %>% anti_join(stop_words)

# removing number
token_title$word <- gsub('[[:digit:]]+', '', token_title$word)

# removing punctuation
token_title$word <- gsub('[[:punct:]]+', '', token_title$word)

# removing edition, mastering, learning, fourth, hands, guide, cookbook, quick, start
token_title$word <- gsub('edition', '', token_title$word)
token_title$word <- gsub('mastering', '', token_title$word)
token_title$word <- gsub('learning', '', token_title$word)
token_title$word <- gsub('fourth', '', token_title$word)
token_title$word <- gsub('hands', '', token_title$word)
token_title$word <- gsub('guide', '', token_title$word)
token_title$word <- gsub('cookbook', '', token_title$word)
token_title$word <- gsub('quick', '', token_title$word)
token_title$word <- gsub('start', '', token_title$word)

# removing where words only contain one character
# programming language such as C or R only have one character, however the requirement already satisfy using the count words
token_title <- token_title %>% filter(!(nchar(word) == 1))%>% anti_join(stop_words)
tokens <- token_title %>% filter(!(word==""))

# bring back to title but with removed stop words and unnecessary word
# tokens <- tokens %>% mutate(ind = row_number())

tokens <- tokens %>% group_by(id) %>% mutate(ind = row_number()) %>%
  tidyr::spread(key = ind, value = word)

tokens [is.na(tokens)] <- ""

tokens <- tidyr::unite(tokens, text,-id,sep =" " )

tokens$text <- trimws(tokens$text)

# combine books based on topic
# creating Document Term Matrix (DTM)
titel_dtm <- CreateDtm(tokens$text, 
                 doc_names = tokens$id, 
                 ngram_window = c(1, 2))

# set a seed so that the output of the model is predictable
# source: https://www.tidytextmining.com/
title_lda <- LDA(titel_dtm, k = 2, control = list(seed = 1234))

# check result
title_topics <- tidy(title_lda, matrix = "beta")

# check top words with only two topic
title_top_terms <- title_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

title_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

##plot the metrics to get number of topics
# source: https://rpubs.com/MNidhi/NumberoftopicsLDA

system.time({
  tunes <- FindTopicsNumber(
    dtm = titel_dtm,
    topics = c(2:15),
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
    method = "Gibbs",
    control = list(seed = 12345),
    mc.cores = 4L,
    verbose = TRUE
  )
})

FindTopicsNumber_plot(tunes)

# Using 5 topics
# set a seed so that the output of the model is predictable
title_lda_5 <- LDA(titel_dtm, k = 5, control = list(seed = 1234))

# check result
title_topics_5 <- tidy(title_lda_5, matrix = "beta")

# check top words with only two topic
title_top_terms_5 <- title_topics_5 %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

title_top_terms_5 %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

# Using 4 topics
# set a seed so that the output of the model is predictable
title_lda_4 <- LDA(titel_dtm, k = 4, control = list(seed = 1234))

# check result
title_topics_4 <- tidy(title_lda_4, matrix = "beta")

# check top words with only two topic
title_top_terms_4 <- title_topics_4 %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

title_top_terms_4 %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

# check which title belong to which topic with probability
# source: https://stackoverflow.com/questions/14875493/lda-with-topicmodels-how-can-i-see-which-topics-different-documents-belong-to
titleDF <- as.data.frame(title_lda_4@gamma) 
names(titleDF) <- c(1:4)
# inspect...
head(titleDF)

# select the column name with the higher probability
topic_number = apply(titleDF,1,which.max)

# assigning tile with topic
tokens$topic_num = topic_number

write.csv(tokens,"title_topic.csv")
