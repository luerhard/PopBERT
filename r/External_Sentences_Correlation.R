# Correlations of PopBERT probabilities 
# with populist example sentences from other studies.

df <- read.csv("./data/Populist_examples_from_other_studies_with_probas.csv")
head(df)

df$Domain
# Binary occurance of Domains
df$Anti.Elite.Source <- c(rep(1, 11), rep(1, 4), 0,0)
df$People.Source <- c(rep(1, 11), rep(1, 2), 0,0, 1,1)

# Correlations
cor(df$Anti.Elite, df$Anti.Elite.Source)
cor(df$People.Centric, df$People.Source)


# Thresholds
#thresh = {"elite": 0.5013018, "centr": 0.5017193, "left": 0.42243505, "right": 0.38281676}

df$Anti.Elite.Prediciton <- ifelse(df$Anti.Elite>=0.5013018, 1, 0)
df$People.Centric.Prediction <- ifelse(df$People.Centric>=0.5017193, 1, 0)

df$Any.Predicted <- ifelse((df$Anti.Elite.Prediciton==1) | (df$People.Centric.Prediction==1), 1,0)

#https://www.r-bloggers.com/2011/06/example-8-39-calculating-cramers-v/
cv.test = function(x,y) {
  CV = sqrt(chisq.test(x, y, correct=FALSE)$statistic /
              (length(x) * (min(length(unique(x)),length(unique(y))) - 1)))
  print.noquote("Cramér V / Phi:")
  return(as.numeric(CV))
}

cv.test(df$People.Centric.Prediction, df$People.Source)
cv.test(df$Anti.Elite.Prediciton, df$Anti.Elite.Source)

df$Snippet[df$Any.Predicted==0]
