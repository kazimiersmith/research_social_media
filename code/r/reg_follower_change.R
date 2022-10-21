data <- 'C:/Users/kas1112/Documents/research_social_media/data'
out <- paste0(data, '/out')
reg_data <- paste0(out, '/estimation')

df_ig = read.csv(paste0(reg_data, '/transition_estimation_data.csv'))

model <- lm(change_log_followers ~ log_frac_spon + log_engagement, data = df_ig)

model
