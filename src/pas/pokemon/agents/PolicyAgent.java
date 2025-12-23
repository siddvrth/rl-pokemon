package src.pas.pokemon.agents;


// SYSTEM IMPORTS
import net.sourceforge.argparse4j.inf.Namespace;

import edu.bu.pas.pokemon.agents.NeuralQAgent;
import edu.bu.pas.pokemon.agents.senses.SensorArray;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.nn.Model;
import edu.bu.pas.pokemon.nn.models.Sequential;
import edu.bu.pas.pokemon.nn.layers.Dense; // fully connected layer
import edu.bu.pas.pokemon.nn.layers.ReLU;  // some activations (below too)
import edu.bu.pas.pokemon.nn.layers.Tanh;
import edu.bu.pas.pokemon.nn.layers.Sigmoid;


// JAVA PROJECT IMPORTS

import src.pas.pokemon.senses.CustomSensorArray;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.Type;
import edu.bu.pas.pokemon.core.SwitchMove;

import java.util.List;
import java.util.Random;


public class PolicyAgent
    extends NeuralQAgent
{
    private Random random;
    private double temperature;
    private double tempDecayRate;
    private double minTemp;
    private boolean trainingMode;
    private int switchCount;

    public PolicyAgent()
    {
        super();
        this.random = new Random();
        this.minTemp = 0.1;
        this.temperature = this.minTemp;
        this.tempDecayRate = 0.999;
        this.trainingMode = false;
        this.switchCount = 0;
    }

    public void initializeSenses(Namespace args)
    {
        SensorArray modelSenses = new CustomSensorArray();

        this.setSensorArray(modelSenses);
    }

    @Override
    public void initialize(Namespace args)
    {
        super.initialize(args);

        this.initializeSenses(args);
    }

    @Override
    public Model initModel()
    {

        Sequential qFunction = new Sequential();

        qFunction.add(new Dense(45, 64));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(64, 64));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(64, 1));

        return qFunction;
    }

    @Override
    public void train() {
        this.trainingMode = true;
        if (this.temperature < 0.5) {
            this.temperature = 1.0;
        }
    }

    @Override
    public void eval() {
        this.trainingMode = false;
    }

    @Override
    public Integer chooseNextPokemon(BattleView view)
    {

        int myTeamIdx = this.getMyTeamIdx();
        TeamView myTeam = view.getTeamView(myTeamIdx);
        TeamView enemyTeam = view.getTeamView(1 - myTeamIdx);

        if (myTeam == null) {
            return null;
        }

        PokemonView enemyPokemon = enemyTeam != null ? enemyTeam.getActivePokemonView() : null;

        Integer bestPokemonIdx = null;
        double bestMatchupScore = Double.NEGATIVE_INFINITY;

        for (int idx = 0; idx < myTeam.size(); idx++) {

            PokemonView p = myTeam.getPokemonView(idx);
            if (p != null && !p.hasFainted()) {

                double matchupEval = evalMatchup(p, enemyPokemon);
                if (matchupEval > bestMatchupScore) {
                    bestMatchupScore = matchupEval;
                    bestPokemonIdx = idx;
                }
            }
        }

        return bestPokemonIdx;
    }

    private double evalMatchup(PokemonView ourPokemon, PokemonView enemyPokemon) {
        double quality = 0.0;

        double healthPercentage = (double) ourPokemon.getCurrentStat(Stat.HP) / 
                                   Math.max(1, ourPokemon.getInitialStat(Stat.HP));
        quality += healthPercentage * 100.0;

        if (enemyPokemon != null) {

            double teamSpeed = ourPokemon.getCurrentStat(Stat.SPD);
            double enemySpeed = enemyPokemon.getCurrentStat(Stat.SPD);

            if (teamSpeed > enemySpeed) {
                quality += 20.0;
            }

            Type enemyType1 = enemyPokemon.getCurrentType1();
            Type enemyType2 = enemyPokemon.getCurrentType2();

            for (MoveView move : ourPokemon.getAvailableMoves()) {
                if (move != null && move.getCategory() != edu.bu.pas.pokemon.core.Move.Category.STATUS) {
                    Type moveType = move.getType();
                    double effectiveness = Type.getEffectivenessModifier(moveType, enemyType1);
                    if (enemyType2 != null) {
                        effectiveness *= Type.getEffectivenessModifier(moveType, enemyType2);
                    }
                    if (effectiveness >= 2.0) {
                        quality += 30.0;
                        break;
                    }
                }
            }
        }

        return quality;
    }


    @Override
    public MoveView getMove(BattleView view)
    {

        if (this.getSensorArray() instanceof CustomSensorArray) {
            ((CustomSensorArray) this.getSensorArray()).setTeamIdx(this.getMyTeamIdx());
        }
        
        List<MoveView> availableMoves = getPotentialMoves(view);

        if (availableMoves == null || availableMoves.isEmpty()) {
            return null;
        }

        if (availableMoves.size() == 1) {
            trackSwitchBehavior(availableMoves.get(0));
            return availableMoves.get(0);
        }

        double[] qValues = new double[availableMoves.size()];

        double maxQValue = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < availableMoves.size(); i++) {
            qValues[i] = eval(view, availableMoves.get(i));
            
            if (availableMoves.get(i) instanceof SwitchMove.SwitchMoveView) {
                if (switchCount >= 2) {
                    qValues[i] -= 1000.0;
                }
            }
            
            if (qValues[i] > maxQValue) {
                maxQValue = qValues[i];
            }
        }

        MoveView chosenMove;
        
        if (trainingMode) {
            double[] expScores = new double[availableMoves.size()];
            double sumExpScores = 0.0;

            for (int i = 0; i < availableMoves.size(); i++) {
                expScores[i] = Math.exp((qValues[i] - maxQValue) / temperature);
                sumExpScores += expScores[i];
            }

            double[] moveProbabilities = new double[availableMoves.size()];
            for (int i = 0; i < availableMoves.size(); i++) {
                moveProbabilities[i] = expScores[i] / sumExpScores;
            }

            double randomValue = random.nextDouble();
            double cumProbability = 0.0;

            chosenMove = availableMoves.get(availableMoves.size() - 1);

            for (int i = 0; i < availableMoves.size(); i++) {
                cumProbability += moveProbabilities[i];
                if (randomValue <= cumProbability) {
                    chosenMove = availableMoves.get(i);
                    break;
                }
            }

        } else {

            int bestMoveIdx = 0;

            for (int i = 1; i < availableMoves.size(); i++) {

                if (qValues[i] > qValues[bestMoveIdx]) {
                    bestMoveIdx = i;}
            }
            chosenMove = availableMoves.get(bestMoveIdx);
            
            MoveView bestAttack = findStrongestAttack(view, availableMoves);

            if (chosenMove instanceof SwitchMove.SwitchMoveView && bestAttack != null) {
                double attackValue = eval(view, bestAttack);
                if (qValues[bestMoveIdx] < attackValue + 750.0) {
                    chosenMove = bestAttack;
                }
            }
        }

        trackSwitchBehavior(chosenMove);
        return chosenMove;
    }

    private void trackSwitchBehavior(MoveView move) {

        if (move instanceof SwitchMove.SwitchMoveView) {
            switchCount++;
        } else {
            switchCount = 0;
        }
    }

    private MoveView findStrongestAttack(BattleView view, List<MoveView> pMoves) {
        int myTeamIdx = this.getMyTeamIdx();

        TeamView myTeam = view.getTeamView(myTeamIdx);
        TeamView enemyTeam = view.getTeamView(1 - myTeamIdx);

        if (myTeam == null || enemyTeam == null) {
            return null;
        }

        PokemonView myActivePokemon = myTeam.getActivePokemonView();
        PokemonView enemyActivePokemon = enemyTeam.getActivePokemonView();

        if (myActivePokemon == null || enemyActivePokemon == null) {
            return null;
        }

        MoveView strongestAttack = null;
        double highestDamage = Double.NEGATIVE_INFINITY;

        Type enemyPrimaryType = enemyActivePokemon.getCurrentType1();
        Type enemySecondaryType = enemyActivePokemon.getCurrentType2();
        Type myPrimaryType = myActivePokemon.getCurrentType1();
        Type mySecondaryType = myActivePokemon.getCurrentType2();

        for (MoveView move : pMoves) {
            if (move instanceof SwitchMove.SwitchMoveView) {
                continue;
            }

            if (move.getCategory() == edu.bu.pas.pokemon.core.Move.Category.STATUS) {
                continue;
            }

            double basePower = move.getPower() != null ? move.getPower() : 0.0;
            if (basePower <= 0) {
                continue;
            }

            Type attackType = move.getType();
            double typeEffectiveness = Type.getEffectivenessModifier(attackType, enemyPrimaryType);
            if (enemySecondaryType != null) {
                typeEffectiveness *= Type.getEffectivenessModifier(attackType, enemySecondaryType);
            }

            if (typeEffectiveness == 0) {
                continue;
            }

            double estimatedDamage = basePower * typeEffectiveness;
            if (attackType == myPrimaryType || (mySecondaryType != null && attackType == mySecondaryType)) {
                estimatedDamage *= 1.5;
            }

            double hitChance = move.getAccuracy() != null ? move.getAccuracy() / 100.0 : 1.0;
            estimatedDamage *= hitChance;

            if (estimatedDamage > highestDamage) {
                highestDamage = estimatedDamage;
                strongestAttack = move;
            }
        }

        return strongestAttack;
    }

    @Override
    public void afterGameEnds(BattleView view)
    {
        switchCount = 0;
        if (trainingMode) {
            temperature = Math.max(minTemp, temperature * tempDecayRate);
        }
    }

}