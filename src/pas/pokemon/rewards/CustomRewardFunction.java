package src.pas.pokemon.rewards;


// SYSTEM IMPORTS


// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.rewards.RewardFunction;
import edu.bu.pas.pokemon.agents.rewards.RewardFunction.RewardType;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.core.SwitchMove;
import edu.bu.pas.pokemon.core.enums.Stat;


public class CustomRewardFunction
    extends RewardFunction
{

    public CustomRewardFunction()
    {
        super(RewardType.STATE_ACTION_STATE); // currently configured to produce rewards as a function of the state
    }

    public double getLowerBound()
    {

        return -1500.0;
    }

    public double getUpperBound()
    {
        return 1500.0;
    }

    public double getStateReward(final BattleView state)
    {
        return evaluateBattleState(state);
    }

    public double getStateActionReward(final BattleView state,
                                       final MoveView action)
    {
        double reward = evaluateBattleState(state);
        if (action instanceof SwitchMove.SwitchMoveView) {reward += -15.0;}
        return reward;
    }

    public double getStateActionStateReward(final BattleView state,
                                            final MoveView action,
                                            final BattleView nextState)
    {
        double totalReward = 0.0;

        int myTeamIdx = 0;
        int enemyTeamIdx = 1 - myTeamIdx;

        TeamView myPreviousTeam = state.getTeamView(myTeamIdx);
        TeamView enemyPreviousTeam = state.getTeamView(enemyTeamIdx);
        TeamView myCurrentTeam = nextState.getTeamView(myTeamIdx);
        TeamView enemyCurrentTeam = nextState.getTeamView(enemyTeamIdx);

        if (myPreviousTeam == null || enemyPreviousTeam == null || 
            myCurrentTeam == null || enemyCurrentTeam == null) {
            return 0.0;
        }

        int myPreviousAlive = countPokemon(myPreviousTeam);
        int enemyPreviousAlive = countPokemon(enemyPreviousTeam);
        int myCurrentAlive = countPokemon(myCurrentTeam);
        int enemyCurrentAlive = countPokemon(enemyCurrentTeam);

        if (enemyCurrentAlive == 0 && myCurrentAlive > 0) {
            return 1000.0;
        }
        if (myCurrentAlive == 0 && enemyCurrentAlive > 0) {
            return -1000.0;
        }
        if (myCurrentAlive == 0 && enemyCurrentAlive == 0) {
            return 0.0;
        }

        double myPreviousHP = calculateTotalHP(myPreviousTeam);
        double enemyPreviousHP = calculateTotalHP(enemyPreviousTeam);
        double myCurrentHP = calculateTotalHP(myCurrentTeam);
        double enemyCurrentHP = calculateTotalHP(enemyCurrentTeam);

        double damageInflicted = enemyPreviousHP - enemyCurrentHP;
        double damageReceived = myPreviousHP - myCurrentHP;

        totalReward += 2.0 * damageInflicted;
        totalReward += -1.0 * damageReceived;

        int enemyKnockouts = enemyPreviousAlive - enemyCurrentAlive;
        int myFaints = myPreviousAlive - myCurrentAlive;

        totalReward += 100.0 * enemyKnockouts;
        totalReward += -100.0 * myFaints;

        if (action instanceof SwitchMove.SwitchMoveView) {
            totalReward += -15.0;
        }

        double myHPRatio = myCurrentHP / Math.max(1, calculateMaxTotalHP(myCurrentTeam));
        double enemyHPRatio = enemyCurrentHP / Math.max(1, calculateMaxTotalHP(enemyCurrentTeam));
        totalReward += 0.5 * (myHPRatio - enemyHPRatio) * 100.0;

        double teamSizeAdvantage = (myCurrentAlive - enemyCurrentAlive) * 10.0;
        totalReward += teamSizeAdvantage;

        return totalReward;
    }

    private double evaluateBattleState(BattleView state) {
        int myTeamIdx = 0;
        int enemyTeamIdx = 1 - myTeamIdx;

        TeamView myTeam = state.getTeamView(myTeamIdx);
        TeamView enemyTeam = state.getTeamView(enemyTeamIdx);

        if (myTeam == null || enemyTeam == null) {
            return 0.0;
        }

        int myAlive = countPokemon(myTeam);
        int enemyAlive = countPokemon(enemyTeam);

        if (enemyAlive == 0 && myAlive > 0) {
            return 1000.0;
        }
        if (myAlive == 0 && enemyAlive > 0) {
            return -1000.0;
        }

        double myTotalHP = calculateTotalHP(myTeam);
        double enemyTotalHP = calculateTotalHP(enemyTeam);
        double myMaxHP = calculateMaxTotalHP(myTeam);
        double enemyMaxHP = calculateMaxTotalHP(enemyTeam);

        double myHPRatio = myTotalHP / Math.max(1, myMaxHP);
        double enemyHPRatio = enemyTotalHP / Math.max(1, enemyMaxHP);

        double reward = (myHPRatio - enemyHPRatio) * 200.0;
        reward += (myAlive - enemyAlive) * 50.0;

        return reward;
    }

    private int countPokemon(TeamView team) {
        int aliveCount = 0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView pokemon = team.getPokemonView(i);
            if (pokemon != null && !pokemon.hasFainted()) {
                aliveCount++;
            }
        }
        return aliveCount;
    }

    private double calculateTotalHP(TeamView team) {
        double totalHP = 0.0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView pokemon = team.getPokemonView(i);
            if (pokemon != null) {
                totalHP += pokemon.getCurrentStat(Stat.HP);
            }
        }
        return totalHP;
    }

    private double calculateMaxTotalHP(TeamView team) {
        double maxTotalHP = 0.0;
        for (int i = 0; i < team.size(); i++) {
            PokemonView pokemon = team.getPokemonView(i);
            if (pokemon != null) {
                maxTotalHP += pokemon.getInitialStat(Stat.HP);
            }
        }
        return maxTotalHP;
    }

}