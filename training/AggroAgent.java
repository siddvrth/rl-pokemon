package edu.bu.pas.pokemon.agents;


// SYSTEM IMPORTS
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


// JAVA PROJECT IMPORTS
import edu.bu.pas.pokemon.core.Agent;
import edu.bu.pas.pokemon.core.Battle;
import edu.bu.pas.pokemon.core.Battle.BattleView;
import edu.bu.pas.pokemon.core.enums.*;
import edu.bu.pas.pokemon.core.Move;
import edu.bu.pas.pokemon.core.Move.Category;
import edu.bu.pas.pokemon.core.Move.MoveView;
import edu.bu.pas.pokemon.core.Pokemon;
import edu.bu.pas.pokemon.core.Pokemon.PokemonView;
import edu.bu.pas.pokemon.core.StruggleMove;
import edu.bu.pas.pokemon.core.SwitchMove;
import edu.bu.pas.pokemon.core.Team;
import edu.bu.pas.pokemon.core.Team.TeamView;
import edu.bu.pas.pokemon.utils.Pair;


public class AggroAgent
    extends Agent
{


    @Override
    public Integer chooseNextPokemon(BattleView rootView)
    {
        int myTeamIdx = this.getMyTeamIdx();
        int oppTeamIdx = (this.getMyTeamIdx() + 1) % 2;

        TeamView myTeam = rootView.getTeamView(myTeamIdx);
        TeamView oppTeam = rootView.getTeamView(oppTeamIdx);

        // put out the most pokemon that can do the most damage to the currently active pokemon on the opps team
        PokemonView oppActivePokemon = oppTeam.getActivePokemonView();

        Integer newPokemonIdx = null;
        double hardestHittingPower = Double.NEGATIVE_INFINITY;
        for(int idx = 0; idx < myTeam.size(); ++idx)
        {
            if(!myTeam.getPokemonView(idx).hasFainted())
            {
                // how much damage can you do?
                double maxMovePower = Double.NEGATIVE_INFINITY;
                for(MoveView availableMove : myTeam.getPokemonView(idx).getAvailableMoves())
                {
                    if(availableMove.getCategory() == Category.PHYSICAL
                       || availableMove.getCategory() == Category.SPECIAL)
                    {
                        if(availableMove.getPower() != null)
                        {
                            maxMovePower = Math.max(maxMovePower, availableMove.getPower());
                        }
                    }
                }

                if(newPokemonIdx == null || maxMovePower > hardestHittingPower)
                {
                    newPokemonIdx = idx;
                    hardestHittingPower = maxMovePower;
                }
            }
        }

        return newPokemonIdx;
    }

    @Override
    public MoveView getMove(BattleView rootView)
    {
        int myTeamIdx = this.getMyTeamIdx();
        int oppTeamIdx = (this.getMyTeamIdx() + 1) % 2;

        TeamView myTeam = rootView.getTeamView(myTeamIdx);
        PokemonView myPokemon = myTeam.getActivePokemonView();

        // how much damage can you do?
        double maxPower = Double.NEGATIVE_INFINITY;
        MoveView maxPowerMove = null;
        for(MoveView availableMove : myPokemon.getAvailableMoves())
        {
            if(availableMove.getCategory() == Category.PHYSICAL
               || availableMove.getCategory() == Category.SPECIAL)
            {
                if(availableMove.getPower() != null && availableMove.getPower() > maxPower)
                {
                    maxPowerMove = availableMove;
                    maxPower = availableMove.getPower();
                }
            }
        }
        if(maxPowerMove == null)
        {
            // choose randomly
            List<MoveView> moves = myPokemon.getAvailableMoves();
            maxPowerMove = moves.get(new Random().nextInt(moves.size()));
        }

        return maxPowerMove;
    }

    @Override
    public void afterGameEnds(BattleView view) {}

}
