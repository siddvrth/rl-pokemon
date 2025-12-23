package edu.bu.pas.pokemon;


// SYSTEM IMPORTS
import java.io.IOException;
import java.io.PrintStream;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;


import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;


// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.NeuralQAgent;
import edu.bu.pas.pokemon.agents.RandomAgent;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction.RewardType;
import edu.bu.pas.pokemon.core.Agent;
import edu.bu.pas.pokemon.core.CoreRegistry;
import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Team;
import edu.bu.pas.pokemon.core.Pokemon;
import edu.bu.pas.pokemon.generators.BattleCreator;
import edu.bu.pas.pokemon.training.data.Dataset;
import edu.bu.pas.pokemon.training.data.ReplacementType;
import edu.bu.pas.pokemon.training.data.ReplayBuffer;

import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.nn.LossFunction;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.nn.Optimizer;
import edu.bu.pas.pokemon.nn.Parameter;
import edu.bu.pas.pokemon.nn.layers.*;
import edu.bu.pas.pokemon.nn.losses.MeanSquaredError;
import edu.bu.pas.pokemon.nn.models.Sequential;
import edu.bu.pas.pokemon.nn.optimizers.*;
import edu.bu.pas.pokemon.utils.Triple;
import edu.bu.pas.pokemon.utils.Pair;


public class Train
    extends Object
{


    public static Agent getAgent(String agentClassName)
    {
        Agent agent = null;
        try
        {
            Class<?> clazz = Class.forName(agentClassName);
            Constructor<?> constructor = clazz.getConstructor();

            agent = (Agent)constructor.newInstance();
        } catch(Exception e)
        {
            System.err.println("[ERROR] Train.getAgent: error when instantiating " + agentClassName);
            e.printStackTrace();
            System.exit(-1);
        }
        return agent;
    }

    public static RewardFunction getRewardFunction()
    {
        RewardFunction rewardFunction = null;
        try
        {
            Class<?> clazz = Class.forName("src.pas.pokemon.rewards.CustomRewardFunction");
            Constructor<?> constructor = clazz.getConstructor();

            rewardFunction = (RewardFunction)constructor.newInstance();
        } catch(Exception e)
        {
            System.err.println("[ERROR] Train.getRewardFunction: error when instantiating "
                + "src.pas.pokemon.rewards.CustomRewardFunction");
            e.printStackTrace();
            System.exit(-1);
        }
        return rewardFunction;
    }

    public static Pair<Optimizer, LossFunction> getModelAdjustmentInfrastructure(Namespace args,
                                                                                 Model model)
    {
        final double lr = args.get("lr");
        final double clipValue = args.get("clip");

        Optimizer optim = null;
        if(args.get("optimizerType").equals("sgd"))
        {
            optim = new SGDOptimizer(
                model.getParameters(),
                lr,
                -clipValue,
                +clipValue
            );
        } else if(args.get("optimizerType").equals("adam"))
        {
            optim = new AdamOptimizer(model.getParameters(),
                                      lr,
                                      args.get("beta1"),
                                      args.get("beta2"),
                                      -clipValue,
                                      +clipValue);
        } else
        {
            System.err.println("[ERROR] Main.getModelAdjustmentInfrastructure: unknown optimizer type "
                + args.get("optimizerType"));
            System.exit(-1);
        }
        LossFunction lossFunction = new MeanSquaredError();

        return new Pair<Optimizer, LossFunction>(optim, lossFunction);
    }

    public static void playTrainingGames(NeuralQAgent agent,
                                         List<Agent> enemyAgents,
                                         ReplayBuffer buffer,
                                         Namespace args,
                                         Random rng)
    {
        final long numTrainingGames = args.get("numTrainingGames");

        try
        {
            for(int gameIdx = 0; gameIdx < numTrainingGames; ++gameIdx)
            {

                for(Agent enemyAgent : enemyAgents)
                {

                    // currently this will make a random battle
                    Battle battle = BattleCreator.makeRandomTeams(6, 6, 4, rng, agent, enemyAgent);

                    // play the game and update the replay buffer
                    try
                    {
                        BattleView oldView = null;
                        MoveView oldAction = null;

                        boolean isGameOver = false;
                        while(!isGameOver)
                        {
                            battle.nextTurn();
                            battle.applyPreTurnConditions();

                            BattleView newView = battle.getView();

                            Pair<Move, Move> moves = battle.getMoves();
                            MoveView action = moves.getFirst().getView();
                            battle.applyMoves(moves);

                            battle.applyPostTurnConditions();

                            newView = battle.getView();
                            isGameOver = battle.isOver();

                            if(oldView != null)
                            {
                                buffer.addSample(oldView, oldAction, newView);
                            }

                            oldView = newView;
                            oldAction = action;
                        }

                        // do afterGameEnds for every agent
                        agent.afterGameEnds(battle.getView());
                        enemyAgent.afterGameEnds(battle.getView());

                        // add transition from nonterminal to terminal state
                        buffer.addSample(oldView, oldAction, battle.getView());

                        // add terminal state
                        // buffer.addSample(battle.getView(), null, null);

                    } catch(Exception e)
                    {
                        // something went wrong in this game
                        e.printStackTrace();
                    }
                }
            }
        } catch(IOException e)
        {
            e.printStackTrace();
        }
    }

    public static void update(NeuralQAgent agent,
                              Optimizer optim,
                              LossFunction lossFunction,
                              ReplayBuffer replayBuffer,
                              RewardFunction rewardFunction,
                              Namespace args,
                              Random rng)
    {
        double discountFactor = args.get("gamma");
        int batchSize = args.get("miniBatchSize");
        int numUpdates = args.get("numUpdates");

        Dataset dataset = replayBuffer.toDataset(agent, discountFactor, rewardFunction);

        // System.err.println("[INFO] Dataset size: X=" + dataset.getFeatures().getShape() + " YGt="
        //     + dataset.getGroundTruth().getShape());

        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            // System.out.println("updating");
            dataset.shuffle();
            Dataset.BatchIterator it = dataset.iterator(batchSize);
            while(it.hasNext())
            {
                Pair<Matrix, Matrix> batch = it.next();

                try
                {
                    Matrix YHat = agent.getModel().forward(batch.getFirst());

                    optim.reset();
                    agent.getModel().backwards(batch.getFirst(),
                                               lossFunction.backwards(YHat, batch.getSecond()));
                    optim.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }            
            }
        }

    }

    public static Pair<Double, Double> playEvalGames(NeuralQAgent agent,
                                                     List<Agent> enemyAgents,
                                                     RewardFunction rewardFunction,
                                                     Namespace args,
                                                     Random rng)
    {
        final long numEvalGames = args.get("numEvalGames");
        final double discountFactor = args.get("gamma");

        double trajectoryUtilitySum = 0;
        double numWins = 0;

        try
        {

            for(int gameIdx = 0; gameIdx < numEvalGames; ++gameIdx)
            {

                for(Agent enemyAgent : enemyAgents)
                {

                    double trajectoryUtility = 0;

                    // currently this will make a random battle
                    Battle battle = BattleCreator.makeRandomTeams(6, 6, 4, rng, agent, enemyAgent);

                    // play the game and compute the expected trajectory
                    try
                    {
                        BattleView oldView = null;
                        MoveView oldAction = null;

                        boolean isGameOver = false;
                        int t = 0;
                        while(!isGameOver)
                        {
                            battle.nextTurn();
                            battle.applyPreTurnConditions();

                            BattleView newView = battle.getView();

                            Pair<Move, Move> moves = battle.getMoves();
                            MoveView action = moves.getFirst().getView();
                            battle.applyMoves(moves);

                            battle.applyPostTurnConditions();

                            newView = battle.getView();
                            isGameOver = battle.isOver();

                            if(oldView != null)
                            {
                                // calculate reward
                                double reward = 0d;

                                switch(rewardFunction.getType())
                                {
                                    case STATE:
                                        reward = rewardFunction.getStateReward(oldView);
                                        break;
                                    case STATE_ACTION:
                                        reward = rewardFunction.getStateActionReward(oldView, oldAction);
                                        break;
                                    case STATE_ACTION_STATE:
                                        reward = rewardFunction.getStateActionStateReward(oldView, oldAction, newView);
                                        break;
                                    default:
                                        System.err.println("[ERROR] Train.playEvalGames: unknown reward function type="
                                            + rewardFunction.getType());
                                        System.exit(1);
                                        break;
                                }

                                trajectoryUtility += Math.pow(discountFactor, t) * reward;
                                t += 1;
                            }

                            oldView = newView;
                            oldAction = action;
                        }

                        // do afterGameEnds for every agent
                        agent.afterGameEnds(battle.getView());
                        enemyAgent.afterGameEnds(battle.getView());

                        // add transition from nonterminal to terminal state
                        // calculate reward
                        double reward = 0d;

                        switch(rewardFunction.getType())
                        {
                            case STATE:
                                reward = rewardFunction.getStateReward(oldView);
                                break;
                            case STATE_ACTION:
                                reward = rewardFunction.getStateActionReward(oldView, oldAction);
                                break;
                            case STATE_ACTION_STATE:
                                reward = rewardFunction.getStateActionStateReward(oldView, oldAction, battle.getView());
                                break;
                            default:
                                System.err.println("[ERROR] Train.playEvalGames: unknown reward function type="
                                    + rewardFunction.getType());
                                System.exit(1);
                                break;
                        }
                        trajectoryUtility += Math.pow(discountFactor, t) * reward;

                        // add terminal state
                        // buffer.addSample(battle.getView(), null, null);

                    } catch(Exception e)
                    {
                        // something went wrong in this game
                        e.printStackTrace();
                    }

                    trajectoryUtilitySum += trajectoryUtility;

                    // who won?
                    boolean team1HasAPokemonLeft = false;
                    for(Pokemon p : battle.getTeam1().getPokemon())
                    {
                        if(p != null)
                        {
                            team1HasAPokemonLeft = team1HasAPokemonLeft || !p.hasFainted();
                        }
                    }

                    boolean team2HasAPokemonLeft = false;
                    for(Pokemon p : battle.getTeam2().getPokemon())
                    {
                        if(p != null)
                        {
                            team2HasAPokemonLeft = team2HasAPokemonLeft || !p.hasFainted();
                        }
                    }

                    if(team1HasAPokemonLeft && !team2HasAPokemonLeft)
                    {
                        numWins += 1;
                    }
                }

            }
        } catch(IOException e)
        {
            e.printStackTrace();
        }

        return new Pair<Double, Double>(trajectoryUtilitySum / numEvalGames, numWins / numEvalGames);
    }


    public static void main(String[] args)
    {
        // overwrite stdout cause there will be a TON of printouts if you don't
        PrintStream out = System.out;
        System.setOut(new PrintStream(out)
            {
                @Override
                public void println(String message) {}
                @Override
                public void print(String message) {}
            }
        );

        ArgumentParser parser = ArgumentParsers.newFor("Train").build()
            .defaultHelp(true)
            .description("Train a NeuralQAgent");

        // agent config
        // TODO: add agent list to compete against
        parser.addArgument("enemyAgents")
            .nargs("*")
            .help("List of enemy agent classpaths you want to train against");
        /**
        parser.addArgument("-q", "--qFunction")
            .type(String.class)
            .setDefault("src.pas.tetris.agents.TetrisQAgent")
            .help("The q-function agent to train if the -a (--agent) argument has been set to " +
                  "edu.bu.tetris.agents.TrainerAgent and ignored otherwise.");
        **/

        // training/eval phase config
        parser.addArgument("-p", "--numCycles")
            .type(Long.class)
            .setDefault(1l)
            .help("the number of times the training/testing cycle is repeated");
        parser.addArgument("-t", "--numTrainingGames")
            .type(Long.class)
            .setDefault(10l)
            .help("the number of training games to collect training data from before an evaluation phase");
        parser.addArgument("-v", "--numEvalGames")
            .type(Long.class)
            .setDefault(5l)
            .help("the number of evaluation games to play while fixing the agent " +
                  "(the agent can't learn from these games)");

        // replay buffer config
        parser.addArgument("-b", "--maxBufferSize")
            .type(Integer.class)
            .setDefault(1280)
            .help("The max number of samples to store in the replay buffer if using the TrainerAgent.");
        parser.addArgument("-r", "--replacementType")
            .type(ReplacementType.class)
            .setDefault(ReplacementType.RANDOM)
            .help("replay buffer replacement type for when a new sample is added to a full buffer");

        // neural network training hyperparams
        parser.addArgument("-u", "--numUpdates")
            .type(Integer.class)
            .setDefault(1)
            .help("the number of epochs to train for after each training phase if using the TrainerAgent.");
        parser.addArgument("-m", "--miniBatchSize")
            .type(Integer.class)
            .setDefault(128)
            .help("batch size to use when performing an epoch of training if using the TrainerAgent.");
        parser.addArgument("-n", "--lr")
            .type(Double.class)
            .setDefault(1e-6)
            .help("the learning rate to use if using the TrainerAgent.");
        parser.addArgument("-c", "--clip")
            .type(Double.class)
            .setDefault(100d)
            .help("gradient clip value to use (symmetric) if using the TrainerAgent.");
        parser.addArgument("-d", "--optimizerType")
            .type(String.class)
            .setDefault("sgd")
            .help("type of optimizer to use if using the TrainerAgent");
        parser.addArgument("-b1", "--beta1")
            .type(Double.class)
            .setDefault(0.9)
            .help("beta1 value for adam optimizer");
        parser.addArgument("-b2", "--beta2")
            .type(Double.class)
            .setDefault(0.999)
            .help("beta2 value for adam optimizer");

        // RL hyperparams
        parser.addArgument("-g", "--gamma")
            .type(Double.class)
            .setDefault(1e-4)
            .help("discount factor for the Bellman equation if using the TrainerAgent.");

        // model saving/loading config
        parser.addArgument("-i", "--inFile")
            .type(String.class)
            .setDefault("")
            .help("params file to load");
        parser.addArgument("-o", "--outFile")
            .type(String.class)
            .setDefault("./params/qFunction")
            .help("where to save the model to (will append XX.model where XX is the number of training/eval " +
                  "cycles performed if using the TrainerAgent.");
        parser.addArgument("--outOffset")
            .type(Long.class)
            .setDefault(1l)
            .help("offset to XX value appended to end of --outFile arg. Useful if you want to resume training from " +
                  "a previous training point and don't want to overwrite any subsequent files. (XX + offset) will " +
                  "be used instead of (XX) when appending to the --outFile arg. Only used if using the TrainerAgent.");

        // miscellaneous config
        parser.addArgument("--seed")
                .type(Long.class)
                .setDefault(-1l)
                .help("random seed to make successive runs repeatable. If -1l, no seed is used");
        


        // parse
        Namespace ns = parser.parseArgsOrFail(args);

        // if you want to see what the namespace looks like they print this
        // out.println(ns);

        // good idea to extract required args
        final long numCycles = ns.get("numCycles");
        final long numEvalGames = ns.get("numEvalGames");

        final long seed = ns.get("seed");

        String checkpointFileBase = ns.get("outFile");
        long offset = ns.get("outOffset");

        // make random seed
        Random rng = new Random(seed);

        // allocate our agent and reward function
        NeuralQAgent agent = (NeuralQAgent)getAgent("src.pas.pokemon.agents.PolicyAgent");
        agent.initialize(ns);
        RewardFunction rewardFunction = getRewardFunction();

        // create enemy agent(s)
        // if you want to make a lineup of enemy agents to play against you could do that here
        List<Agent> enemyAgents = new LinkedList<>();
        List<String> enemyAgentClassPaths = ns.get("enemyAgents");
        if(enemyAgentClassPaths.size() == 0)
        {
            System.err.println("[ERROR] Train.main: need to specify at least one enemy agent to train against!");
            System.exit(1);
        } else
        {
            for(String classPath : enemyAgentClassPaths)
            {
                Agent enemyAgent = getAgent(classPath);
                enemyAgent.initialize(ns);
                enemyAgents.add(enemyAgent);
            }
        }

        // some training machinery
        Pair<Optimizer, LossFunction> adjustmentPair = getModelAdjustmentInfrastructure(
            ns,
            agent.getModel()
        );
        Optimizer optim = adjustmentPair.getFirst();
        LossFunction lossFunction = adjustmentPair.getSecond();

        ReplayBuffer replayBuffer = new ReplayBuffer(
            ns.get("replacementType"),
            ns.get("maxBufferSize"),
            rng
        );

        for(int cycleIdx = 0; cycleIdx < numCycles; ++cycleIdx)
        {
            agent.train();
            playTrainingGames(agent, enemyAgents, replayBuffer, ns, rng);

            // update the model
            update(agent, optim, lossFunction, replayBuffer, rewardFunction, ns, rng);

            // save the model
            agent.getModel().save(checkpointFileBase + (cycleIdx + offset) + ".model");

            agent.eval();
            Pair<Double, Double> statsPair = playEvalGames(agent, enemyAgents, rewardFunction, ns, rng);
            double avgUtil = statsPair.getFirst();
            double avgNumWins = statsPair.getSecond();

            // use the variable out to actually write to console
            out.println("after cycle=" + cycleIdx + " avg(utility)=" + avgUtil + " avg(num_wins)=" + avgNumWins);
        }
    }
}

