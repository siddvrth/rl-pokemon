package src.pas.pokemon.senses;


// SYSTEM IMPORTS


// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.agents.senses.SensorArray;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.linalg.Matrix;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.SwitchMove;
import edu.bu.pas.pokemon.core.enums.Stat;
import edu.bu.pas.pokemon.core.enums.Type;
import edu.bu.pas.pokemon.core.enums.NonVolatileStatus;


public class CustomSensorArray
    extends SensorArray
{
    
    private int currentTeamIdx = 0;

    public CustomSensorArray(){ }

    public void setTeamIdx(int idx) {
        this.currentTeamIdx = idx;
    }

    public Matrix getSensorValues(final BattleView state, final MoveView action)
    {

        Matrix featureVector = Matrix.zeros(1, 45);
        int featureIdx = 0;

        try {

            TeamView myTeam = state.getTeamView(this.currentTeamIdx);
            TeamView enemyTeam = state.getTeamView(1 - this.currentTeamIdx);

            if (myTeam == null || enemyTeam == null) {
                return featureVector;
            }

            PokemonView myCurrPokemon = myTeam.getActivePokemonView();
            PokemonView enemyCurrPokemon = enemyTeam.getActivePokemonView();

            if (myCurrPokemon == null || enemyCurrPokemon == null) {
                return featureVector;
            }

            double myMaxHP = Math.max(1, myCurrPokemon.getInitialStat(Stat.HP));
            double myCurrentHP = myCurrPokemon.getCurrentStat(Stat.HP);
            featureVector.set(0, featureIdx++, myCurrentHP / myMaxHP);

            double enemyMaxHP = Math.max(1, enemyCurrPokemon.getInitialStat(Stat.HP));
            double enemyCurrentHP = enemyCurrPokemon.getCurrentStat(Stat.HP);

            featureVector.set(0, featureIdx++, enemyCurrentHP / enemyMaxHP);

            featureVector.set(0, featureIdx++, myCurrPokemon.getCurrentStat(Stat.ATK) / 400.0);
            featureVector.set(0, featureIdx++, myCurrPokemon.getCurrentStat(Stat.DEF) / 400.0);
            featureVector.set(0, featureIdx++, myCurrPokemon.getCurrentStat(Stat.SPD) / 400.0);
            featureVector.set(0, featureIdx++, myCurrPokemon.getCurrentStat(Stat.SPATK) / 400.0);
            featureVector.set(0, featureIdx++, myCurrPokemon.getCurrentStat(Stat.SPDEF) / 400.0);

            featureVector.set(0, featureIdx++, enemyCurrPokemon.getCurrentStat(Stat.ATK) / 400.0);
            featureVector.set(0, featureIdx++, enemyCurrPokemon.getCurrentStat(Stat.DEF) / 400.0);
            featureVector.set(0, featureIdx++, enemyCurrPokemon.getCurrentStat(Stat.SPD) / 400.0);
            featureVector.set(0, featureIdx++, enemyCurrPokemon.getCurrentStat(Stat.SPATK) / 400.0);
            featureVector.set(0, featureIdx++, enemyCurrPokemon.getCurrentStat(Stat.SPDEF) / 400.0);

            boolean weAreFirstToMove = myCurrPokemon.getCurrentStat(Stat.SPD) >= 
                                        enemyCurrPokemon.getCurrentStat(Stat.SPD);
            featureVector.set(0, featureIdx++, weAreFirstToMove ? 1.0 : 0.0);

            int aliveCount = 0;
            for (int i = 0; i < myTeam.size(); i++) {
                PokemonView pokemon = myTeam.getPokemonView(i);
                if (pokemon != null && !pokemon.hasFainted()) {
                    aliveCount++;
                }
            }
            featureVector.set(0, featureIdx++, aliveCount / 6.0);

            int enemyAliveCount = 0;
            for (int i = 0; i < enemyTeam.size(); i++) {
                PokemonView pokemon = enemyTeam.getPokemonView(i);
                if (pokemon != null && !pokemon.hasFainted()) {
                    enemyAliveCount++;
                }
            }
            featureVector.set(0, featureIdx++, enemyAliveCount / 6.0);

            featureVector.set(0, featureIdx++, (aliveCount - enemyAliveCount) / 6.0);

            boolean isSwitch = action instanceof SwitchMove.SwitchMoveView;
            featureVector.set(0, featureIdx++, isSwitch ? 1.0 : 0.0);

            if (!isSwitch && action != null) {

                double movePower = action.getPower() != null ? action.getPower() : 0.0;
                featureVector.set(0, featureIdx++, movePower / 250.0);

                double moveAccuracy = action.getAccuracy() != null ? action.getAccuracy() : 100.0;
                featureVector.set(0, featureIdx++, moveAccuracy / 100.0);

                featureVector.set(0, featureIdx++, action.getPriority() / 5.0);

                Move.Category moveCategory = action.getCategory();

                featureVector.set(0, featureIdx++, moveCategory == Move.Category.PHYSICAL ? 1.0 : 0.0);
                featureVector.set(0, featureIdx++, moveCategory == Move.Category.SPECIAL ? 1.0 : 0.0);
                featureVector.set(0, featureIdx++, moveCategory == Move.Category.STATUS ? 1.0 : 0.0);

                Type moveType = action.getType();

                Type myPrimaryType = myCurrPokemon.getCurrentType1();
                Type mySecondaryType = myCurrPokemon.getCurrentType2();
                boolean sameTypeAttackBonus = (moveType == myPrimaryType) || 
                                               (mySecondaryType != null && moveType == mySecondaryType);
                featureVector.set(0, featureIdx++, sameTypeAttackBonus ? 1.0 : 0.0);

                Type enemyPrimaryType = enemyCurrPokemon.getCurrentType1();
                Type enemySecondaryType = enemyCurrPokemon.getCurrentType2();

                double typeEffectiveness = Type.getEffectivenessModifier(moveType, enemyPrimaryType);

                if (enemySecondaryType != null) {
                    typeEffectiveness *= Type.getEffectivenessModifier(moveType, enemySecondaryType);
                }

                featureVector.set(0, featureIdx++, typeEffectiveness / 4.0);
                featureVector.set(0, featureIdx++, typeEffectiveness >= 2.0 ? 1.0 : 0.0);
                featureVector.set(0, featureIdx++, typeEffectiveness <= 0.5 ? 1.0 : 0.0);
                featureVector.set(0, featureIdx++, typeEffectiveness == 0.0 ? 1.0 : 0.0);


//https://bulbapedia.bulbagarden.net/wiki/Damage#Generation_I

                double expectedDamage = 0.0;

                if (moveCategory == Move.Category.PHYSICAL && movePower > 0) {

                    double attackStat = myCurrPokemon.getCurrentStat(Stat.ATK);
                    double defenseStat = Math.max(1, enemyCurrPokemon.getCurrentStat(Stat.DEF));

                    expectedDamage = (((2.0 * 100.0 / 5.0 + 2.0) * movePower * attackStat / defenseStat) / 50.0 + 2.0);

                } else if (moveCategory == Move.Category.SPECIAL && movePower > 0) {

                    double specialAttackStat = myCurrPokemon.getCurrentStat(Stat.SPATK);
                    double specialDefenseStat = Math.max(1, enemyCurrPokemon.getCurrentStat(Stat.SPDEF));

                    expectedDamage = (((2.0 * 100.0 / 5.0 + 2.0) * movePower * specialAttackStat / specialDefenseStat) / 50.0 + 2.0);
                }

                expectedDamage *= typeEffectiveness;

                if (sameTypeAttackBonus) {

                    expectedDamage *= 1.5;
                }
                featureVector.set(0, featureIdx++, expectedDamage / 500.0);

                boolean canOneHitKO = expectedDamage >= enemyCurrentHP;
                featureVector.set(0, featureIdx++, canOneHitKO ? 1.0 : 0.0);

                double damageRatio = expectedDamage / Math.max(1, enemyCurrentHP);
                featureVector.set(0, featureIdx++, Math.min(damageRatio, 2.0) / 2.0);

            } else {

                featureIdx += 14;
            }



            NonVolatileStatus status = myCurrPokemon.getNonVolatileStatus();

            featureVector.set(0, featureIdx++, status != NonVolatileStatus.NONE ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, status == NonVolatileStatus.PARALYSIS ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, status == NonVolatileStatus.BURN ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, status == NonVolatileStatus.POISON || 
                                               status == NonVolatileStatus.TOXIC ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, status == NonVolatileStatus.SLEEP ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, status == NonVolatileStatus.FREEZE ? 1.0 : 0.0);

            NonVolatileStatus enemyStatus = enemyCurrPokemon.getNonVolatileStatus();
            
            featureVector.set(0, featureIdx++, enemyStatus != NonVolatileStatus.NONE ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, enemyStatus == NonVolatileStatus.PARALYSIS ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, enemyStatus == NonVolatileStatus.SLEEP ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, enemyStatus == NonVolatileStatus.FREEZE ? 1.0 : 0.0);

            featureVector.set(0, featureIdx++, myTeam.getNumLightScreenTurnsRemaining() > 0 ? 1.0 : 0.0);
            featureVector.set(0, featureIdx++, myTeam.getNumReflectTurnsRemaining() > 0 ? 1.0 : 0.0);

        } catch (Exception e) {System.out.println("Error");}

        return featureVector;
    }

    public int getNumFeatures() {
        return 45;
    }

}