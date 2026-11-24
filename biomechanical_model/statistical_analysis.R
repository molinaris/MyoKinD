# KINEMATIC STATISTICAL ANALYSIS
#
# Statistical analysis of participant-level kinematic metrics.
#
# Input:
#   kinematic_metrics_participants.csv
#
# Outputs:
#   kinematic_statistical_analysis.txt
#   kinematic_significant_interactions.csv
#
# For each metric and finger:
#   - Shapiro-Wilk test for each Task × Frequency condition
#   - LMM if all conditions satisfy normality
#   - ART otherwise
#   - Type III tests for Frequency, Task, and Task × Frequency
#   - Bonferroni-corrected Task post-hoc comparisons
#   - Bonferroni-corrected frequency-within-Task comparisons
#     when the Task × Frequency interaction is significant
#
# Input and output files are expected in the same directory
# as this script.


suppressPackageStartupMessages({
  library(tidyverse)
  library(lmerTest)
  library(pbkrtest)
  library(ARTool)
  library(emmeans)})

options(contrasts = c("contr.sum", "contr.poly"), width = 180)
emmeans::emm_options(lmer.df = "kenward-roger")


METRICS <- c("rom_deg", "mean_speed_deg_s", "thdn_percent")

METRIC_LABELS <- c(rom_deg = "RoM", mean_speed_deg_s = "Mean Speed", thdn_percent = "THDN")

FINGERS <- c("thumb", "index", "middle", "ring", "little")

FINGER_LABELS <- c(thumb = "Thumb", index = "Index", middle = "Middle",
                   ring = "Ring", little = "Little")

TASK_LEVELS <- as.character(1:8)
FREQUENCY_LEVELS <- c("05", "075")


get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)

  if (length(file_arg))
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/")))

  for (frame in rev(sys.frames()))
    if (!is.null(frame$ofile))
      return(dirname(normalizePath(frame$ofile, winslash = "/")))

  normalizePath(getwd(), winslash = "/")}


sig_code <- function(p) {
  case_when(is.na(p) ~ "", p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ "")}


format_p <- function(p) {
  ifelse(is.na(p), "NA", ifelse(p < 0.001, "< 0.001", sprintf("%.3f", p)))}


pretty_effect <- function(x) {
  key <- tolower(gsub("\\s+", "", x))
  result <- x

  result[key == "task"] <- "Task"
  result[key == "frequency"] <- "Frequency"
  result[key %in% c("task:frequency", "frequency:task")] <- "Task × Frequency"

  result}


print_progress <- function(done, total, metric, finger) {
  width <- 30
  fraction <- done / total
  filled <- floor(width * fraction)
  bar <- paste0(strrep("#", filled), strrep("-", width - filled))

  cat(sprintf("\r[%s] %d/%d (%3.0f%%)  %s - %s",
    bar, done, total, 100 * fraction, metric, finger))

  flush.console()
  if (done == total) cat("\n")}


# PATHS
SCRIPT_DIR <- get_script_dir()

input_file <- file.path(SCRIPT_DIR, "kinematic_metrics_participants.csv")
stats_output_file <- file.path(SCRIPT_DIR, "kinematic_statistical_analysis.txt")
figure_output_file <- file.path(SCRIPT_DIR, "kinematic_significant_interactions.csv")


report <- function(...) {cat(..., file = stats_output_file, append = TRUE, sep = "")}

report_table <- function(x) {
  output <- capture.output(print(x, row.names = FALSE))
  report(paste(output, collapse = "\n"), "\n")}


# DATA
if (!file.exists(input_file)) stop("Input file not found: ", input_file)

df <- readr::read_csv(
  input_file,
  show_col_types = FALSE,
  col_types = cols(participant = col_character(), task = col_character(),
                   frequency = col_character(), finger = col_character(),
                   rom_deg = col_double(), mean_speed_deg_s = col_double(),
                   thdn_percent = col_double()))

required_columns <- c("participant", "task", "frequency", "finger",
                       "rom_deg", "mean_speed_deg_s", "thdn_percent")

missing_columns <- setdiff(required_columns, names(df))

if (length(missing_columns)) {stop("Missing columns: ", paste(missing_columns, collapse = ", "))}

df <- df %>%
  mutate(
    participant = factor(participant),
    task = factor(task, levels = TASK_LEVELS),
    frequency = factor(frequency, levels = FREQUENCY_LEVELS),
    finger = factor(finger, levels = FINGERS))

participants_used <- sort(unique(as.character(df$participant)))
participants_string <- paste(participants_used, collapse = "|")


# REPORT FILE
if (file.exists(stats_output_file)) invisible(file.remove(stats_output_file))

report(
  "KINEMATIC STATISTICAL ANALYSIS\n",
  "====================================================================\n",
  "Participants: ", paste(participants_used, collapse = ", "), "\n",
  "Parametric: LMM | Type III F tests | Kenward-Roger df\n",
  "Non-parametric: ART mixed-effects model\n",
  "Post-hoc correction: Bonferroni\n",
  "Significance: * p < 0.05 | ** p < 0.01 | *** p < 0.001\n",
  "====================================================================\n")


# ANALYSIS
all_significant_interactions <- list()
total_steps <- length(METRICS) * length(FINGERS)
current_step <- 0

for (metric in METRICS) {
  for (f in FINGERS) {
    current_step <- current_step + 1
    print_progress(current_step, total_steps, METRIC_LABELS[[metric]], FINGER_LABELS[[f]])

    df_f <- df %>% filter(finger == f) # Finger data
    tasks <- levels(droplevels(df_f$task))
    freqs <- levels(droplevels(df_f$frequency))

    # Descriptive statistics and Shapiro-Wilk
    summary_list <- list()

    for (t in tasks) {
      for (fr in freqs) {
        subset_data <- df_f %>% filter(task == t, frequency == fr)
        values <- subset_data[[metric]]
        values <- values[!is.na(values)]
        n_obs <- length(values)

        if (n_obs >= 3) shapiro_p <- shapiro.test(values)$p.value else shapiro_p <- NA_real_
        if (n_obs > 0) mean_value <- mean(values) else mean_value <- NA_real_
        if (n_obs >= 2) sd_value <- sd(values) else sd_value <- NA_real_

        mean_sd <- if (!is.na(mean_value) && !is.na(sd_value))
          sprintf("%.2f ± %.2f", mean_value, sd_value) else "NA"

        summary_list[[paste(t, fr, sep = "_")]] <- data.frame(
          Task = t,
          Frequency = fr,
          n = n_obs,
          `Mean ± SD` = mean_sd,
          Shapiro_p_raw = shapiro_p,
          Shapiro_p = format_p(shapiro_p),
          Normal = ifelse(!is.na(shapiro_p) & shapiro_p >= 0.05, "YES", "NO"),
          check.names = FALSE)}}

    descriptive_table <- bind_rows(summary_list)
    all_normal <- all(!is.na(descriptive_table$Shapiro_p_raw)
                      & descriptive_table$Shapiro_p_raw >= 0.05)
    descriptive_output <- descriptive_table %>% select(-Shapiro_p_raw)

    # Linear mixed-effects model
    if (all_normal) {
      analysis_type <- "LMM"
      mixed_model <- lmerTest::lmer(
        as.formula(paste(metric, "~ task * frequency + (1 | participant)")),
        data = df_f, REML = TRUE)
      mixed_anova <- anova(mixed_model, type = 3, ddf = "Kenward-Roger")
      mixed_raw <- as.data.frame(mixed_anova)
      effect_raw <- rownames(mixed_raw)
      p_raw <- mixed_raw$`Pr(>F)`
      eta2_partial <- with(mixed_raw, `F value` * NumDF / (`F value` * NumDF + DenDF))

      omnibus_table <- data.frame(
        Effect = pretty_effect(effect_raw),
        df1 = round(mixed_raw$NumDF, 3),
        df2 = round(mixed_raw$DenDF, 3),
        F = round(mixed_raw$`F value`, 3),
        p = format_p(p_raw),
        eta2_partial = round(eta2_partial, 3),
        Sig = sig_code(p_raw),
        check.names = FALSE)
    } else {
      # Aligned Rank Transform
      analysis_type <- "ART"
      art_model <- ARTool::art(
        as.formula(paste(metric, "~ task * frequency + (1 | participant)")),
        data = df_f)
      art_anova <- anova(art_model)
      if ("Term" %in% names(art_anova)) effect_raw <- as.character(art_anova$Term)
      else effect_raw <- rownames(art_anova)
      p_raw <- art_anova$`Pr(>F)`
      eta2_partial <- with(art_anova, F * Df / (F * Df + Df.res))

      omnibus_table <- data.frame(
        Effect = pretty_effect(effect_raw),
        df1 = round(art_anova$Df, 3),
        df2 = round(art_anova$Df.res, 3),
        F = round(art_anova$F, 3),
        p = format_p(p_raw),
        eta2_partial = round(eta2_partial, 3),
        Sig = sig_code(p_raw),
        check.names = FALSE)}

    omnibus_table <- omnibus_table[match(c("Frequency", "Task", "Task × Frequency"),
                                         omnibus_table$Effect),]
    report("\n\n",
           "====================================================================\n",
           "METRIC: ", METRIC_LABELS[[metric]],
           " | FINGER: ", FINGER_LABELS[[f]], "\n",
           "====================================================================\n")
    report("\n1. DESCRIPTIVE STATISTICS + SHAPIRO-WILK\n",
           "--------------------------------------------------------------------\n")
    report_table(descriptive_output)
    report("\n2. MODEL\n", analysis_type, " | ", metric,
           " ~ task * frequency + (1 | participant)")
    if (analysis_type == "LMM") report(" | Kenward-Roger df")
    report("\n", "\n3. OMNIBUS ANALYSIS\n",
           "--------------------------------------------------------------------\n")
    report_table(omnibus_table)


    # Significant effects
    effect_key <- tolower(gsub("\\s+", "", effect_raw))
    significant_mask <- !is.na(p_raw) & p_raw < 0.05
    interaction_significant <- any(significant_mask &
        effect_key %in% c("task:frequency", "frequency:task"))
    task_significant <- any(significant_mask & effect_key == "task")
    frequency_significant <- any(significant_mask & effect_key == "frequency")

    # Post-hoc
    report("\n4. POST-HOC\n",
           "--------------------------------------------------------------------\n")

    if (interaction_significant) {
      report("Task × Frequency significant -> Frequency within Task\n\n")

      if (analysis_type == "LMM") {
        interaction_emmeans <- suppressMessages(
          emmeans::emmeans(mixed_model, ~ frequency | task, lmer.df = "kenward-roger"))
        ph_interaction <- pairs(interaction_emmeans, adjust = "none") %>%
          as.data.frame()
        ph_interaction$p_bonf <- p.adjust(ph_interaction$p.value, method = "bonferroni")

        interaction_output <- data.frame(
          Task = as.character(ph_interaction$task),
          Comparison = gsub(" - ", " vs ", ph_interaction$contrast),
          estimate = round(ph_interaction$estimate, 3),
          SE = round(ph_interaction$SE, 3),
          df = round(ph_interaction$df, 3),
          t_ratio = round(ph_interaction$t.ratio, 3),
          p_bonf = format_p(ph_interaction$p_bonf),
          Sig = sig_code(ph_interaction$p_bonf),
          check.names = FALSE)

        figure_rows <- ph_interaction %>%
          filter(p_bonf < 0.05) %>%
          transmute(
            Metric = metric,
            Finger = f,
            Task_or_Factor = paste0("Task=", as.character(task)),
            A = "05",
            B = "075",
            p_bonf = p_bonf)
      } else {
        # ART interaction post-hoc
        art_contrasts_all <- ARTool::art.con(art_model, "task:frequency", adjust = "none") %>%
          as.data.frame()

        contrast_parts <- stringr::str_split_fixed(
          as.character(art_contrasts_all$contrast), " - ", 2)

        left_parts <- stringr::str_split_fixed(contrast_parts[, 1], ",", 2)
        right_parts <- stringr::str_split_fixed(contrast_parts[, 2], ",", 2)

        art_contrasts_all$task_A <- trimws(left_parts[, 1])
        art_contrasts_all$freq_A <- trimws(left_parts[, 2])
        art_contrasts_all$task_B <- trimws(right_parts[, 1])
        art_contrasts_all$freq_B <- trimws(right_parts[, 2])

        ph_interaction <- art_contrasts_all %>%
          filter(task_A == task_B, freq_A != freq_B) %>%
          arrange(as.numeric(task_A))

        if (nrow(ph_interaction) == 0) {
          stop(paste("Could not identify ART-C frequency-within-Task contrasts for",
                  metric, "-", f))}
        ph_interaction$p_bonf <- p.adjust(ph_interaction$p.value, method = "bonferroni")

        interaction_output <- data.frame(
          Task = ph_interaction$task_A,
          Comparison = paste0(ph_interaction$freq_A, " vs ", ph_interaction$freq_B),
          estimate = round(ph_interaction$estimate, 3),
          SE = round(ph_interaction$SE, 3),
          df = round(ph_interaction$df, 3),
          t_ratio = round(ph_interaction$t.ratio, 3),
          p_bonf = format_p(ph_interaction$p_bonf),
          Sig = sig_code(ph_interaction$p_bonf),
          check.names = FALSE)

        figure_rows <- ph_interaction %>%
          filter(p_bonf < 0.05) %>%
          transmute(
            Metric = metric,
            Finger = f,
            Task_or_Factor = paste0("Task=", task_A),
            A = "05",
            B = "075",
            p_bonf = p_bonf)}

      report_table(interaction_output)

      if (nrow(figure_rows) > 0)
        all_significant_interactions[[length(all_significant_interactions) + 1]] <- figure_rows
    } else {
      # No significant interaction
      if (task_significant) report("Task significant -> pairwise Task comparisons\n")
      else report("No Task post-hoc comparisons required\n")

      if (frequency_significant) {
        report("Frequency has two levels; ",
               "the omnibus test provides the 0.50 vs 0.75 Hz comparison directly.\n")}
      if (task_significant) report("\n")

      # Task post-hoc
      if (task_significant) {
        if (analysis_type == "LMM") {
          ph_task <- suppressMessages(emmeans::emmeans(mixed_model, ~ task, lmer.df = "kenward-roger")) %>%
            pairs(adjust = "bonferroni") %>%
            as.data.frame()
        } else ph_task <- ARTool::art.con(art_model, "task", adjust = "bonferroni") %>% as.data.frame()

        task_output <- data.frame(
          Comparison = gsub(" - ", " vs ", ph_task$contrast),
          estimate = round(ph_task$estimate, 3),
          SE = round(ph_task$SE, 3),
          df = round(ph_task$df, 3),
          t_ratio = round(ph_task$t.ratio, 3),
          p_bonf = format_p(ph_task$p.value),
          Sig = sig_code(ph_task$p.value),
          check.names = FALSE)

        report("--- Task ---\n")
        report_table(task_output)}}
  }
}

# SIGNIFICANCE CSV FOR PYTHON
if (length(all_significant_interactions) > 0) {
  final_posthoc <- bind_rows(all_significant_interactions) %>%
    mutate(
      Metric = factor(Metric, levels = METRICS),
      Finger = factor(Finger, levels = FINGERS),
      task_order = as.numeric(sub("Task=", "", Task_or_Factor))
    ) %>%
    arrange(Metric, Finger, task_order) %>%
    mutate(Metric = as.character(Metric), Finger = as.character(Finger)) %>%
    select(-task_order)
} else {
  final_posthoc <- data.frame(
    Metric = NA_character_,
    Finger = NA_character_,
    Task_or_Factor = NA_character_,
    A = NA_character_,
    B = NA_character_,
    p_bonf = NA_real_)}

final_posthoc$Participants <- participants_string
write.csv(final_posthoc, figure_output_file, row.names = FALSE)


cat("\nAnalysis completed successfully.\n",
    "Participants: ", paste(participants_used, collapse = ", "), "\n",
    "Statistical output: ",
    normalizePath(stats_output_file, winslash = "/", mustWork = FALSE), "\n",
    "Significance CSV: ",
    normalizePath(figure_output_file, winslash = "/", mustWork = FALSE), "\n",
    sep = "")

