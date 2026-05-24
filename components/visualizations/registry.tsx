import type { ComponentType } from "react";

import { AttentionFlow } from "@/components/visualizations/attention-is-all-you-need";
import { ConservativePolicyIteration } from "@/components/visualizations/approximately-optimal-approximate-reinforcement-learning";
import { QLearningGrid } from "@/components/visualizations/learning-from-delayed-rewards";
import { PolicyGradientGrid } from "@/components/visualizations/policy-gradient-methods";
import { GaussianSearch } from "@/components/visualizations/statistical-gradient-following";
import { AtariReplay } from "@/components/visualizations/playing-atari-with-deep-reinforcement-learning";
import { TrustRegionFlow } from "@/components/visualizations/trust-region-policy-optimization";
import { GeneralizedAdvantageEstimation } from "@/components/visualizations/high-dimensional-continuous-control-using-gae";
import { ClippedSurrogate } from "@/components/visualizations/proximal-policy-optimization";
import { CrazyStoneSearch } from "@/components/visualizations/efficient-selectivity-and-backup-operators";
import { GoTreeSearch } from "@/components/visualizations/mastering-the-game-of-go";
import { MonteCarloTreeSearch } from "@/components/visualizations/mastering-chess-and-shogi-by-self-play";
import { ExpertIterationLoop } from "@/components/visualizations/thinking-fast-and-slow-with-deep-learning-and-tree-search";
import { Seq2SeqRelay } from "@/components/visualizations/sequence-to-sequence-learning-with-neural-networks";
import { AlignAndTranslate } from "@/components/visualizations/neural-machine-translation-by-jointly-learning-to-align-and-translate";
import { ZeroShotGenerator } from "@/components/visualizations/language-models-are-unsupervised-multitask-learners";
import { LanguageModelsAreFewShotLearners } from "@/components/visualizations/language-models-are-few-shot-learners";
import { ScalingLawsExplorer } from "@/components/visualizations/scaling-laws";
import { PreferenceLoop } from "@/components/visualizations/deep-reinforcement-learning-from-human-preferences";
import { TrainingWithHumanFeedback } from "@/components/visualizations/training-language-models-to-follow-instructions-with-human-feedback";
import { DirectPreferenceOptimization } from "@/components/visualizations/direct-preference-optimization";
import { ChainOfThought } from "@/components/visualizations/chain-of-thought-prompting";
import { GroupRelativePolicyOptimization } from "@/components/visualizations/deepseekmath-pushing-the-limits";
import { ImplicitRewardTraining } from "@/components/visualizations/process-reinforcement-through-implicit-rewards";
import { ReActLoop } from "@/components/visualizations/react-synergizing-reasoning-and-acting";
import { ReflexionLoop } from "@/components/visualizations/reflexion-language-agents-with-verbal-reinforcement-learning";
import { TreeOfThoughtsSearch } from "@/components/visualizations/tree-of-thoughts-deliberate-problem-solving";
import { VoyagerLifelongLoop } from "@/components/visualizations/voyager-an-open-ended-embodied-agent";
import { WebArenaAgent } from "@/components/visualizations/webarena";
import { OSWorldAgentLoop } from "@/components/visualizations/osworld";
import { CrmArenaAgent } from "@/components/visualizations/crmarena";
import { TurnLevelCredit } from "@/components/visualizations/reinforcing-multi-turn-reasoning-in-llm-agents";
import { GradientGeometry } from "@/components/visualizations/on-sft-rl-and-on-policy-distillation";
import { AgentQSearch } from "@/components/visualizations/agent-q-advanced-reasoning-and-learning";
import { VisualizationBoundary } from "@/components/visualizations/visualization-boundary";

// Maps a paper slug to its interactive visualization. The MDX renderer looks up
// the match and injects it under the name <Visualization />. Each entry is
// wrapped in the error boundary so a crash in its loop shows a fallback card
// instead of taking down the page. Adding a paper with a visual adds one import
// and one entry here (see the add-paper skill).
const visualizations: Record<string, ComponentType> = {
  "attention-is-all-you-need": () => (
    <VisualizationBoundary>
      <AttentionFlow />
    </VisualizationBoundary>
  ),
  "learning-from-delayed-rewards": () => (
    <VisualizationBoundary>
      <QLearningGrid />
    </VisualizationBoundary>
  ),
  "statistical-gradient-following": () => (
    <VisualizationBoundary>
      <GaussianSearch />
    </VisualizationBoundary>
  ),
  "policy-gradient-methods": () => (
    <VisualizationBoundary>
      <PolicyGradientGrid />
    </VisualizationBoundary>
  ),
  "approximately-optimal-approximate-reinforcement-learning": () => (
    <VisualizationBoundary>
      <ConservativePolicyIteration />
    </VisualizationBoundary>
  ),
  "playing-atari-with-deep-reinforcement-learning": () => (
    <VisualizationBoundary>
      <AtariReplay />
    </VisualizationBoundary>
  ),
  "trust-region-policy-optimization": () => (
    <VisualizationBoundary>
      <TrustRegionFlow />
    </VisualizationBoundary>
  ),
  "high-dimensional-continuous-control-using-gae": () => (
    <VisualizationBoundary>
      <GeneralizedAdvantageEstimation />
    </VisualizationBoundary>
  ),
  "proximal-policy-optimization": () => (
    <VisualizationBoundary>
      <ClippedSurrogate />
    </VisualizationBoundary>
  ),
  "efficient-selectivity-and-backup-operators": () => (
    <VisualizationBoundary>
      <CrazyStoneSearch />
    </VisualizationBoundary>
  ),
  "mastering-the-game-of-go": () => (
    <VisualizationBoundary>
      <GoTreeSearch />
    </VisualizationBoundary>
  ),
  "mastering-chess-and-shogi-by-self-play": () => (
    <VisualizationBoundary>
      <MonteCarloTreeSearch />
    </VisualizationBoundary>
  ),
  "thinking-fast-and-slow-with-deep-learning-and-tree-search": () => (
    <VisualizationBoundary>
      <ExpertIterationLoop />
    </VisualizationBoundary>
  ),
  "sequence-to-sequence-learning-with-neural-networks": () => (
    <VisualizationBoundary>
      <Seq2SeqRelay />
    </VisualizationBoundary>
  ),
  "neural-machine-translation-by-jointly-learning-to-align-and-translate":
    () => (
      <VisualizationBoundary>
        <AlignAndTranslate />
      </VisualizationBoundary>
    ),
  "language-models-are-unsupervised-multitask-learners": () => (
    <VisualizationBoundary>
      <ZeroShotGenerator />
    </VisualizationBoundary>
  ),
  "language-models-are-few-shot-learners": () => (
    <VisualizationBoundary>
      <LanguageModelsAreFewShotLearners />
    </VisualizationBoundary>
  ),
  "scaling-laws": () => (
    <VisualizationBoundary>
      <ScalingLawsExplorer />
    </VisualizationBoundary>
  ),
  "deep-reinforcement-learning-from-human-preferences": () => (
    <VisualizationBoundary>
      <PreferenceLoop />
    </VisualizationBoundary>
  ),
  "training-language-models-to-follow-instructions-with-human-feedback": () => (
    <VisualizationBoundary>
      <TrainingWithHumanFeedback />
    </VisualizationBoundary>
  ),
  "direct-preference-optimization": () => (
    <VisualizationBoundary>
      <DirectPreferenceOptimization />
    </VisualizationBoundary>
  ),
  "chain-of-thought-prompting": () => (
    <VisualizationBoundary>
      <ChainOfThought />
    </VisualizationBoundary>
  ),
  "deepseekmath-pushing-the-limits": () => (
    <VisualizationBoundary>
      <GroupRelativePolicyOptimization />
    </VisualizationBoundary>
  ),
  "process-reinforcement-through-implicit-rewards": () => (
    <VisualizationBoundary>
      <ImplicitRewardTraining />
    </VisualizationBoundary>
  ),
  "react-synergizing-reasoning-and-acting": () => (
    <VisualizationBoundary>
      <ReActLoop />
    </VisualizationBoundary>
  ),
  "reflexion-language-agents-with-verbal-reinforcement-learning": () => (
    <VisualizationBoundary>
      <ReflexionLoop />
    </VisualizationBoundary>
  ),
  "tree-of-thoughts-deliberate-problem-solving": () => (
    <VisualizationBoundary>
      <TreeOfThoughtsSearch />
    </VisualizationBoundary>
  ),
  "voyager-an-open-ended-embodied-agent": () => (
    <VisualizationBoundary>
      <VoyagerLifelongLoop />
    </VisualizationBoundary>
  ),
  webarena: () => (
    <VisualizationBoundary>
      <WebArenaAgent />
    </VisualizationBoundary>
  ),
  osworld: () => (
    <VisualizationBoundary>
      <OSWorldAgentLoop />
    </VisualizationBoundary>
  ),
  crmarena: () => (
    <VisualizationBoundary>
      <CrmArenaAgent />
    </VisualizationBoundary>
  ),
  "reinforcing-multi-turn-reasoning-in-llm-agents": () => (
    <VisualizationBoundary>
      <TurnLevelCredit />
    </VisualizationBoundary>
  ),
  "on-sft-rl-and-on-policy-distillation": () => (
    <VisualizationBoundary>
      <GradientGeometry />
    </VisualizationBoundary>
  ),
  "agent-q-advanced-reasoning-and-learning": () => (
    <VisualizationBoundary>
      <AgentQSearch />
    </VisualizationBoundary>
  ),
};

export function getVisualization(slug: string): ComponentType | null {
  return visualizations[slug] ?? null;
}
