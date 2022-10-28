library(stargazer)

data <- 'C:/Users/kas1112/Documents/research_social_media/data'
out <- paste0(data, '/out')
reg_data <- paste0(out, '/estimation')

df_ig = read.csv(paste0(reg_data, '/transition_estimation_data.csv'))

model <- lm(change_log_followers ~ log_frac_spon + log_engagement, data = df_ig)
summary(model)

model2 <- lm(change_log_followers ~ log_frac_spon + log_engagement + log_posts, data = df_ig)
summary(model2)

model3 <- lm(change_log_followers ~ log_frac_spon + log_engagement + log_posts + log_likes, data = df_ig)
summary(model3)

model4 <- lm(change_log_followers ~ log_frac_spon + log_engagement + log_posts + log_comments, data = df_ig)
summary(model4)

model5 <- lm(change_log_followers ~ log_frac_spon + log_engagement + log_posts + log_likes + log_comments, data = df_ig)
summary(model5)

stargazer(model, model2, model3, model4, model5, out = 'results.tex')

          