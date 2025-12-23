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
import edu.bu.pas.pokemon.execution.BattlePool;
import edu.bu.pas.pokemon.execution.EvalBattleDriver;
import edu.bu.pas.pokemon.execution.OutcomeUtilityResult;
import edu.bu.pas.pokemon.execution.Result;
import edu.bu.pas.pokemon.execution.ThreadPool;
import edu.bu.pas.pokemon.execution.TrainingBattleDriver;
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


public class ParallelTrain
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

    public static void playTrainingGames(Agent agent,
                                         List<Agent> enemyAgents,
                                         ReplayBuffer buffer,
                                         Namespace args,
                                         Random rng,
                                         PrintStream out)
    {

        final int numThreads = args.get("numThreads");
        final long numTrainingGames = args.get("numTrainingGames");
        TrainingBattleDriver driver = new TrainingBattleDriver(buffer);

        BattlePool battlePool = new BattlePool( // iterates over the matchups
            agent,
            enemyAgents,
            numTrainingGames,
            rng,
            driver
        );


        // yes yes I know its slower to be constantly allocating new threads and then killing them at the end
        // and it would be faster to just have these threads persist but I had concurrency trouble getting that
        // to work and didn't feel like debugging it lol
        ThreadPool threadPool = new ThreadPool( // distributes matchups across workers
            battlePool,
            numThreads,
            rng
        );

        // this commented out code was how I originally intended to do it: the ThreadPool itself would be recycled
        // so that the existing threads could just be repurposed between training/eval and across cycles.
        // The way this would work is that you would swap out the BattlePool the ThreadPool is using
        // and the BattlePool could have a replaceable driver.
        // I ended up having some concurrency problems that manifested in the eval loop not playing all of the expected
        // number of games and entering a deadlock which I found too tedious to debug and instead opted for the (slower)
        // solution of just spawning new threads for a fixed BattlePool with a fixed BattleDriver.
        //
        // load the battlePool with the driver responsible for making & running training games
        // TrainingBattleDriver driver = new TrainingBattleDriver(buffer);
        // threadPool.getBattlePool().setBattleDriver(driver);

        // reset progress bar & underlying battle pool
        // threadPool.reset();

        // out.println("[INFO] ParallelTrain.playTrainingGames: numBattles=" + threadPool.getNumTotalBattles() +
        //     " to play");

        try
        {
            threadPool.start();
            while(threadPool.getNumBattlesProcessed() < threadPool.getNumTotalBattles())
            {
                List<Result> results = threadPool.update();
                // out.println("[INFO] ParallelTrain.playTrainingGames: " + results.size() + " battles finished " +
                //     "-> total number processed = " + threadPool.getNumBattlesProcessed());
            }
        }
        catch(Exception e)
        {
            e.printStackTrace();
        } finally
        {
            threadPool.stopNow();
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

    public static Pair<Double, Double> playEvalGames(Agent agent,
                                                     List<Agent> enemyAgents,
                                                     RewardFunction rewardFunction,
                                                     Namespace args,
                                                     Random rng,
                                                     PrintStream out)
    {

        final int numThreads = args.get("numThreads");
        final long numEvalGames = args.get("numEvalGames");
        final double discountFactor = args.get("gamma");

        // yes yes I know its slower to be constantly allocating new threads and then killing them at the end
        // and it would be faster to just have these threads persist but I had concurrency trouble getting that
        // to work and didn't feel like debugging it lol
        EvalBattleDriver driver = new EvalBattleDriver(rewardFunction, discountFactor);

        BattlePool battlePool = new BattlePool(
            agent,
            enemyAgents,
            numEvalGames,
            rng,
            driver
        );

        ThreadPool threadPool = new ThreadPool( // distributes matchups across workers
            battlePool,
            numThreads,
            rng
        );

        double trajectoryUtilitySum = 0;
        double numWins = 0;
        int numGames = 0;

        // this commented out code was how I originally intended to do it: the ThreadPool itself would be recycled
        // so that the existing threads could just be repurposed between training/eval and across cycles.
        // The way this would work is that you would swap out the BattlePool the ThreadPool is using
        // and the BattlePool could have a replaceable driver.
        // I ended up having some concurrency problems that manifested in the eval loop not playing all of the expected
        // number of games and entering a deadlock which I found too tedious to debug and instead opted for the (slower)
        // solution of just spawning new threads for a fixed BattlePool with a fixed BattleDriver.
        //
        // load the battlePool with the driver responsible for making & running training games
        // EvalBattleDriver driver = new EvalBattleDriver(rewardFunction, discountFactor);
        // threadPool.getBattlePool().setBattleDriver(driver);

        // reset progress bar & underlying battle pool
        // threadPool.reset();

        // out.println("[INFO] ParallelTrain.playTrainingGames: numBattles=" + threadPool.getNumTotalBattles() +
        //     " to play");

        try
        {
            threadPool.start();
            while(threadPool.getNumBattlesProcessed() < threadPool.getNumTotalBattles())
            {
                List<Result> results = threadPool.update();
                // out.println("[INFO] ParallelTrain.playTrainingGames: " + results.size() + " battles finished " +
                //     "-> total number processed = " + threadPool.getNumBattlesProcessed());

                for(Result r : results)
                {
                    if(r instanceof OutcomeUtilityResult)
                    {
                        numGames += 1;

                        Integer outcome = ((OutcomeUtilityResult)r).getOutcome();
                        double trajectoryUtility = ((OutcomeUtilityResult)r).getUtility();

                        trajectoryUtilitySum += trajectoryUtility;
                        if(outcome != null && outcome == 0) { numWins += 1; }
                    }
                }
            }
        } catch(Exception e)
        {
            e.printStackTrace();
        } finally
        {
            threadPool.stopNow();
        }

        return new Pair<Double, Double>(
            trajectoryUtilitySum / numGames,
            numWins / numGames
        );
    }


    public static void main(String[] args)
    {
        // overwrite stdout cause there will be a TON of printouts if you don't
        PrintStream out = System.out;

        /****/
        System.setOut(new PrintStream(out)
            {
                @Override
                public void println(String message) {}
                @Override
                public void print(String message) {}
            }
        );
        /****/

        ArgumentParser parser = ArgumentParsers.newFor("ParallelTrain").build()
            .defaultHelp(true)
            .description("Train a NeuralQAgent where training games are played in parallel and " +
                         "eval games are played in parallel.");

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
            .type(int.class)
            .setDefault(1280)
            .help("The max number of samples (e.g. transitions) to store in the replay buffer.");
        parser.addArgument("-r", "--replacementType")
            .type(ReplacementType.class)
            .setDefault(ReplacementType.RANDOM)
            .help("replay buffer replacement type for when a new sample is added to a full buffer");

        // neural network training hyperparams
        parser.addArgument("-u", "--numUpdates")
            .type(int.class)
            .setDefault(1)
            .help("the number of epochs to train for after each training phase.");
        parser.addArgument("-m", "--miniBatchSize")
            .type(int.class)
            .setDefault(128)
            .help("batch size to use when performing an epoch of training.");
        parser.addArgument("-n", "--lr")
            .type(double.class)
            .setDefault(1e-6)
            .help("the learning rate to use.");
        parser.addArgument("-c", "--clip")
            .type(double.class)
            .setDefault(100d)
            .help("gradient clip value to use (symmetric).");
        parser.addArgument("-d", "--optimizerType")
            .type(String.class)
            .setDefault("sgd")
            .help("type of optimizer to use");
        parser.addArgument("-b1", "--beta1")
            .type(double.class)
            .setDefault(0.9)
            .help("beta1 value for adam optimizer");
        parser.addArgument("-b2", "--beta2")
            .type(double.class)
            .setDefault(0.999)
            .help("beta2 value for adam optimizer");

        // RL hyperparams
        parser.addArgument("-g", "--gamma")
            .type(Double.class)
            .setDefault(1e-4)
            .help("discount factor for the Bellman equation.");

        // model saving/loading config
        parser.addArgument("-i", "--inFile")
            .type(String.class)
            .setDefault("")
            .help("params file to load");
        parser.addArgument("-o", "--outFile")
            .type(String.class)
            .setDefault("./params/qFunction")
            .help("where to save the model to (will append XX.model where XX is the number of training/eval " +
                  "cycles performed.");
        parser.addArgument("--outOffset")
            .type(long.class)
            .setDefault(1l)
            .help("offset to XX value appended to end of --outFile arg. Useful if you want to resume training from " +
                  "a previous training point and don't want to overwrite any subsequent files. (XX + offset) will " +
                  "be used instead of (XX) when appending to the --outFile arg.");

        // miscellaneous config
        parser.addArgument("--seed")
                .type(long.class)
                .setDefault(-1l)
                .help("random seed to make successive runs repeatable. If -1l, no seed is used");
        parser.addArgument("-j", "--numThreads")
                .type(int.class)
                .setDefault(1)
                .help("number of worker threads to use (so will use <num_threads> + 1 total threads)");
        


        // parse
        Namespace ns = parser.parseArgsOrFail(args);

        // if you want to see what the namespace looks like they print this
        // out.println(ns);

        // good idea to extract required args
        final long numCycles = ns.get("numCycles");
        final long numTrainingGames = ns.get("numTrainingGames");
        final long numEvalGames = ns.get("numEvalGames");

        final long seed = ns.get("seed");

        String checkpointFileBase = ns.get("outFile");
        long offset = ns.get("outOffset");

        // make random seed
        Random rng = new Random(seed);
        final int numThreads = ns.get("numThreads");

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

        /**
        // I had originally intended to recycle the ThreadPool and just swap the BattlePool that was generating
        // Battle instances on the fly. It would be faster than spawning/killing the same number of threads over and
        // over but I had trouble getting it to work: I encountered a deadlock situation where upon swapping
        // a larger BattlePool for a smaller one (e.g. going from training to eval) the eval pool wouldn't play the
        // expected number of games and hang.
        // Rather than debug that I decided it would be quicker on my part (and yes slower to run) to not share the
        // same thread pool between training/eval. If you want to try and debug this issue (I think its in
        // BattlePool.java and ThreadPool.java) be my guest.

        // now make the parallel infrastructure
        BattlePool trainingBattlePool = new BattlePool( // iterates over the matchups
            agent,
            enemyAgents,
            numTrainingGames,
            rng
        );

        BattlePool evalBattlePool = new BattlePool(
            agent,
            enemyAgents,
            numEvalGames,
            rng
        );

        ThreadPool threadPool = new ThreadPool( // distributes matchups across workers
            numThreads,
            rng
        );
        **/

        for(int cycleIdx = 0; cycleIdx < numCycles; ++cycleIdx)
        {
            agent.train();
            // threadPool.setBattlePool(trainingBattlePool);
            playTrainingGames(agent, enemyAgents, replayBuffer, ns, rng, out);

            // update the model
            update(agent, optim, lossFunction, replayBuffer, rewardFunction, ns, rng);

            // save the model
            agent.getModel().save(checkpointFileBase + (cycleIdx + offset) + ".model");

            agent.eval();
            // threadPool.setBattlePool(evalBattlePool);
            Pair<Double, Double> statsPair = playEvalGames(agent, enemyAgents, rewardFunction, ns, rng, out);
            double avgUtil = statsPair.getFirst();
            double avgNumWins = statsPair.getSecond();

            // use the variable out to actually write to console
            out.println("after cycle=" + cycleIdx + " avg(utility)=" + avgUtil + " avg(num_wins)=" + avgNumWins);
        }
    }
}

