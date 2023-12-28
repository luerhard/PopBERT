#### CHES Validation ####


#### Legislative term 18 ####
# Oct. 2013 - Oct. 2017


#ch14 <- read.csv("./data/CHES/2014_CHES_dataset_means.csv")
#ch14 <- ch14[ch14$cname=="ger", c("party_name","antielite_salience")]
#ch14$people_vs_elite <- NA
#ch14


# CHES 2017

ch17 <- read.csv("./data/CHES/CHES_means_2017.csv")
ch17 <- ch17[ch17$country=="ger", c("party","people_vs_elite","antielite_salience")]
ch17$term <- 18

# weighted average for CDU/CSU based on 255/56 seats:
cducsu_ppl <- ((ch17$people_vs_elite[ch17$party=="CDU"]*255) + (ch17$people_vs_elite[ch17$party=="CSU"] *56)) / (255+56)
cducsu_anti <- ((ch17$antielite_salience[ch17$party=="CDU"]*255) + (ch17$antielite_salience[ch17$party=="CSU"] *56)) / (255+56)

ch17 <- rbind(ch17,
              list('CDU/CSU', 
                   cducsu_ppl,
                   cducsu_anti,
                   18))

# Subset for parties in parliament
ch17 <- ch17[ch17$party %in% c("CDU/CSU","SPD", "Grunen","Linke"),]
ch17$party[ch17$party=="Grunen"] <- "GRUNEN"
ch17$party[ch17$party=="Linke"] <- "LINKE"


#### Legislative term 19 ####
# Oct. 2017 - Oct. 20121

# CHES 2019

ch19 <- read.csv("./data/CHES/CHES2019V3.csv")
ch19 <- ch19[ch19$country==3, c("party","people_vs_elite","antielite_salience")]
ch19$term <- 19


# weighted average for CDU/CSU based on 200 / 45 seats:
cducsu_ppl <- ((ch19$people_vs_elite[ch19$party=="CDU"]*200) + (ch19$people_vs_elite[ch19$party=="CSU"] *45)) / (200+45)
cducsu_anti <- ((ch19$antielite_salience[ch19$party=="CDU"]*200) + (ch19$antielite_salience[ch19$party=="CSU"] *45)) / (200+45)

ch19 <- rbind(ch19,
              list('CDU/CSU', 
                   cducsu_ppl,
                   cducsu_anti,
                   19))

# Subset for parties in parliament
ch19 <- ch19[ch19$party %in% c("CDU/CSU","SPD", "GRUNEN","LINKE", "FDP", "AfD"),]


# Merge CHES 2017 and CHES 2019

df <- rbind(ch17, ch19)
df$term <- as.character(df$term)


#### Import PopBert scales by party and term ####

# Data from figure 1
pbert <- read.csv("./data/figure_1_numbers.csv")


df$party[df$party=="GRUNEN"] <- "Grüne"
df$party[df$party=="LINKE"] <- "DIE LINKE."


df <- merge(df, pbert[pbert$variable=="(a) Anti-Elitism",],
            by.x=c("party", "term"),
            by.y=c("Party", "electoral_term"))[,-5]
names(df)[5] <- "Anti_Elitism"

df <- merge(df, pbert[pbert$variable=="(b) People-Centrism",],
            by.x=c("party", "term"),
            by.y=c("Party", "electoral_term"))[,-6]
names(df)[6] <- "People_Centrism"


cor(df$people_vs_elite , df$People_Centrism)
cor(df$antielite_salience  , df$Anti_Elitism)


df_long <- data.frame(Party=rep(df$party, 2),
                      Term=rep(df$term, 2),
                      Concept=c(rep("Anti-Elitism", nrow(df)), 
                                rep("People-Centrism", nrow(df))),
                      CHES=c(df$people_vs_elite, df$antielite_salience),
                      PopBERT=c(df$Anti_Elitism, df$People_Centrism))


library(ggplot2)
ggplot(df_long, aes(x=CHES, y=PopBERT, 
               color=Term,
               label=Party)) + 
  geom_text() +
  scale_x_continuous(limits=c(0,10)) +
  theme_bw() +
  facet_wrap(.~Concept,
             scales="free_y")+
  theme(legend.position = "bottom")

# With fixed y-axis
ggplot(df_long, aes(x=CHES, y=PopBERT, 
                    color=Term,
                    label=Party)) + 
  geom_text() +
  scale_x_continuous(limits=c(0,10)) +
  theme_bw() +
  facet_wrap(.~Concept,
             scales="fixed")+
  theme(legend.position = "bottom")





