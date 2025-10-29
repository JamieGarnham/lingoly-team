# Mixed Effects Logistic Regression Analysis in R
# Install required packages if needed:
# install.packages(c("tidyverse", "lme4", "broom.mixed"))

library(tidyverse)
library(lme4)
library(broom.mixed)

# Read the data
df <- read.csv('chained_monolingual_LRME_32_llama.csv')
cat("Data loaded successfully!\n")
cat("Number of rows:", nrow(df), "\n")
cat("Number of columns:", ncol(df), "\n\n")

# Step 1: Reshape from wide to long format
cat("Reshaping data to long format...\n")

df_long <- df %>%
  pivot_longer(
    cols = matches('^is_v\\d+_correct$'),
    names_to = 'student',
    values_to = 'is_correct',
    names_pattern = 'is_v(\\d+)_correct'
  ) %>%
  mutate(
    student_id = as.numeric(student),
    # Convert is_first to proper boolean
    is_first = ifelse(is_first == "True" | is_first == TRUE, TRUE, FALSE),
    # Ensure is_correct is logical
    is_correct = as.logical(is_correct)
  ) %>%
  select(exam_id = overall_question_n, student_id, is_first, is_correct, question_n)

cat("Long format created:", nrow(df_long), "observations\n\n")

# Debug: Check the data after transformation
cat("Sample of transformed data:\n")
print(head(df_long))
cat("is_first values:\n")
print(table(df_long$is_first, useNA = "always"))

# Step 2: Get Q1 results for each student/exam combination
q1_results <- df_long %>%
  filter(is_first == TRUE) %>%
  select(exam_id, student_id, q1_correct = is_correct)

cat("Q1 results created:", nrow(q1_results), "observations\n")

# Step 3: Merge Q1 results back and filter to Q2-N questions
df_analysis <- df_long %>%
  left_join(q1_results, by = c('exam_id', 'student_id')) %>%
  filter(is_first == FALSE) %>%
  drop_na(is_correct, q1_correct)

cat("Analysis dataset:", nrow(df_analysis), "observations\n")
cat("Number of unique exams:", n_distinct(df_analysis$exam_id), "\n")
cat("Number of unique students per exam: 32\n\n")

# Convert to factors where appropriate
df_analysis <- df_analysis %>%
  mutate(
    exam_id = factor(exam_id),
    student_id = factor(student_id),
    is_correct = as.numeric(is_correct),
    q1_correct = as.numeric(q1_correct)
  )

cat(paste(rep("=", 60), collapse=""), "\n")
cat("MIXED EFFECTS LOGISTIC REGRESSION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Fit the mixed effects logistic regression model
# Random intercept for exam_id to account for varying exam difficulty
cat("Fitting mixed effects model with random intercept for exam...\n")
cat("(This may take a moment...)\n\n")

model <- glmer(
  is_correct ~ q1_correct + (1 | exam_id),
  data = df_analysis,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)

cat("Model fitting complete!\n\n")

# Print model summary
cat(paste(rep("=", 60), collapse=""), "\n")
cat("MODEL SUMMARY\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")
print(summary(model))

# Extract key statistics
fixed_effects <- fixef(model)
q1_coef <- fixed_effects['q1_correct']
q1_se <- sqrt(diag(vcov(model)))['q1_correct']
q1_z <- q1_coef / q1_se
q1_pval <- 2 * (1 - pnorm(abs(q1_z)))
q1_odds_ratio <- exp(q1_coef)
q1_ci_lower <- exp(q1_coef - 1.96 * q1_se)
q1_ci_upper <- exp(q1_coef + 1.96 * q1_se)

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("KEY RESULTS (controlling for exam difficulty)\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("Fixed Effect for Q1 Correct:\n")
cat(sprintf("  Coefficient (β): %.4f\n", q1_coef))
cat(sprintf("  Standard Error: %.4f\n", q1_se))
cat(sprintf("  Z-value: %.4f\n", q1_z))
cat(sprintf("  P-value: %.6f\n", q1_pval))

cat("\nOdds Ratio:\n")
cat(sprintf("  Odds Ratio: %.4f\n", q1_odds_ratio))
cat(sprintf("  95%% CI: [%.4f, %.4f]\n", q1_ci_lower, q1_ci_upper))

# Determine significance
if (q1_pval < 0.001) {
  sig_level <- "p < 0.001 (highly significant)"
} else if (q1_pval < 0.01) {
  sig_level <- "p < 0.01 (very significant)"
} else if (q1_pval < 0.05) {
  sig_level <- "p < 0.05 (significant)"
} else {
  sig_level <- "p > 0.05 (not significant)"
}

cat(sprintf("\nSignificance: %s\n", sig_level))

# Random effects variance
random_effects <- VarCorr(model)
exam_variance <- as.numeric(random_effects$exam_id[1])
exam_sd <- sqrt(exam_variance)

cat("\nRandom Effects (Exam-level variation):\n")
cat(sprintf("  Variance: %.4f\n", exam_variance))
cat(sprintf("  Standard Deviation: %.4f\n", exam_sd))

# Intraclass Correlation Coefficient (ICC)
# Proportion of variance due to exam differences
icc <- exam_variance / (exam_variance + pi^2/3)
cat(sprintf("  ICC: %.4f (%.1f%% of variance due to exam differences)\n", 
            icc, icc * 100))

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("INTERPRETATION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("After controlling for exam difficulty (random effects):\n")
cat(sprintf("• Students who get Q1 correct have %.2fx the odds of getting\n", q1_odds_ratio))
cat("  Q2-N correct compared to those who get Q1 wrong\n")
cat(sprintf("• We are 95%% confident the true odds ratio is between %.2f and %.2f\n", 
            q1_ci_lower, q1_ci_upper))

# Calculate marginal effects (average predicted probabilities)
# Create prediction datasets
pred_data_q1_incorrect <- data.frame(q1_correct = 0)
pred_data_q1_correct <- data.frame(q1_correct = 1)

# Predict on population level (fixed effects only, averaging over random effects)
# Using the inverse logit function
logit_q1_incorrect <- fixed_effects['(Intercept)']
logit_q1_correct <- fixed_effects['(Intercept)'] + fixed_effects['q1_correct']

prob_q1_incorrect <- plogis(logit_q1_incorrect)
prob_q1_correct <- plogis(logit_q1_correct)

cat("\nAverage predicted probabilities (population level):\n")
cat(sprintf("• When Q1 incorrect: %.1f%%\n", prob_q1_incorrect * 100))
cat(sprintf("• When Q1 correct: %.1f%%\n", prob_q1_correct * 100))
cat(sprintf("• Difference: %.1f percentage points\n", 
            (prob_q1_correct - prob_q1_incorrect) * 100))

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("MODEL COMPARISON\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Fit null model for comparison
model_null <- glmer(
  is_correct ~ 1 + (1 | exam_id),
  data = df_analysis,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)

# Likelihood ratio test
lr_test <- anova(model_null, model)
cat("Likelihood Ratio Test (Q1 effect vs. null model):\n")
print(lr_test)

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("CONCLUSION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

if (q1_pval < 0.001) {
  cat("✓ Getting Q1 correct is a HIGHLY SIGNIFICANT predictor of Q2-N\n")
  cat("  performance, even after controlling for exam difficulty.\n\n")
} else if (q1_pval < 0.05) {
  cat("✓ Getting Q1 correct is a SIGNIFICANT predictor of Q2-N\n")
  cat("  performance, even after controlling for exam difficulty.\n\n")
} else {
  cat("✗ Getting Q1 correct is NOT a significant predictor of Q2-N\n")
  cat("  performance after controlling for exam difficulty.\n\n")
}

cat(sprintf("The odds ratio of %.2f means that answering Q1 correctly\n", q1_odds_ratio))
cat("increases the odds of answering subsequent questions correctly\n")
cat(sprintf("by a factor of %.2f, controlling for exam-specific difficulty.\n", q1_odds_ratio))

# Save results
results <- data.frame(
  Parameter = c("Q1 Correct", "Exam SD (Random Effect)"),
  Coefficient = c(q1_coef, exam_sd),
  Std_Error = c(q1_se, NA),
  P_value = c(q1_pval, NA),
  Odds_Ratio = c(q1_odds_ratio, NA),
  CI_Lower = c(q1_ci_lower, NA),
  CI_Upper = c(q1_ci_upper, NA)
)

write.csv(results, 'mixed_effects_results.csv', row.names = FALSE)
cat("\nResults saved to 'mixed_effects_results.csv'\n")

# Optional: Extract and save random effects by exam
exam_effects <- ranef(model)$exam_id
exam_effects$exam_id <- rownames(exam_effects)
colnames(exam_effects)[1] <- "random_intercept"
write.csv(exam_effects, 'exam_random_effects.csv', row.names = FALSE)
cat("Exam-specific random effects saved to 'exam_random_effects.csv'\n")