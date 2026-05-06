using Godot;
using System;
using System.IO;

public partial class DataManager : Node {

    // ……………………………………//"~-,_)/ _/''\....('_,-~-|' . „„--„

    // ………………………………-~*")"/'~)||||||/'\'\"/'~-, /'||\:,-~"""'\ .'/' '(, ,-~-, \,

    // …………………………….._„-~" |||||||||||||||||||||||'\/'|||||||||||||\\-~"||"~--,/',,--"|//'"|(

    // ………………………………'\;;;(/'|||||||||||||||||||||||||||||||||||||||||||||||||||(,-~"|||||||(__"~-|

    // …………………………__….'\,/||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||-~-„_)

    // ……………………_„„-~/||'\;;;/'|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||("*~-„

    // …………………...(;;///|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'\

    // ……………………///||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||)

    // ………………….||||||||||||||||||||||||||||||||||||\\\\\|||||||||\\\\||||||||||||||||||///|||||||||||||||||||||||||||(

    // ……………….,/"'|||||||||||||||||||||||||||||||/' . . . . . . .\\\\\\\||||////// . . . . \\///// \\||||||||||) ,

    // ……………….|||||||||||||||||||||||||||||||||/' . . . . . . . . . . . . . . . . . . . . . . . . . '\\||||||/ /)

    // ……………..._)||||||||||||||||||||||||||||/' . . . . . . . . . . . . . . . . . . . . . . . . . . . .'\|||||||/

    // ……………,/'|||||||||||||||||||||||||||||||| . . . . . . . . . . . . . . . . . . . . . . . . . . . . . '\|||||(;;;;)

    // ……………(|||||||||||||||||||||||||||||||| . . . . . . . . . . . . ,, . . .~--- . .__ .-~~ ~ . . . '|||||/""

    // …………….."\||||||||||||||||||||||||||||\\;;; . . . . . . .-~' . . . . . . . . . . . . . . . . . . . . \|||(,/'

    // ………………..||||||||||||||||||||||||||||||// . . . . . . . . . . . . . . . . . . . . . . . . . . „„„„\\\|||||)

    // ………………..||||||||||||||||||||||||||||/ . . . . . . . _„„„„„_______ . . . . . . . .:;;||||///~"'||/

    // ………………'\|||//||||||||||||||||||||:: . . . . . . .;;/'||||||||||||||||||||||//; . . . . . . /'""__ . . .'|

    // ………………../' :/""~|||||||||||| . . . . . . :::: . . .__„„„„„__:::;;;; . . . . ;;-~*|*~-„ . /

    // ………………..|.:|: . .:'\|||||||| . . . . . . . _ „-~"-~~*"¯¯ ;;::::: . . . . .;;'*~'~*"';; '||

    // ………………..|.:|:. ./// ||||||| . . . . . . . . . . .' -- -- - ~' ::::::: . . . . .|: .' ~ . .:::;|

    // ………………..'|:'|: ::|\ '|||||| . . . . . . . . . . . . . . . . . . .::: . . . . . .'| . . . . . . .|

    // ………………...'\:'\::: .¯|||||||| .: : : : : :: . . . . . . . . . . . :: . . . . . . . . . . . . . |

    // …………………..'\:: . . '|||||||||| : : : :: :: : :: . . . . . . . .::;;:: .: : . . .| . . . . . |

    // …………………….'\,_ .||||||||||| : : : : : : : : : : : . . . . .( . . .___ . . _ |:: . . .|'

    // ……………………….¯¯¯'\||||||: : : : : : : : : : : . . . . . . ."~"""""~-~"~' :: . ./'           he's an importer-exporter guy 

    // ……………………………/'| : : : : : : : : : : : : : . . . . . . / : : : ;;; : : : : : :'|

    // …………………………,/';;'|:'\ : : : : : : : : : : : : . . . . . .. . . ;;;;;::: : : : : '|

    // ………………………,/';;;;;;'|::'~, : : : : : : : : : : : ;;_„„„___„„„___„„„„---; : :'|a

    // ………………….,-~*;;;;;;;;;|::::::'~, : : : : : : : : :;; : : '~,_ . . . . . . ,/' : :,/''*~-„_g

    // …………_„„„--~";;;;;;;;;;;;;;;|:::::::::: ' , : : : : : : : :.. . . . .¯¯""""""¯ : : /';;;;;;;;;;¯"~-,a

    // …_„„-~"¯;;;;;;;;;;;;;;;;;;;;;;;;;'\:::::::::::::;;'\: : : : : : : : : : : : . . . . . : : /';;;;;;;;;;;;;;;;;;;;"~-,_

    // *¯;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'\::::::: : : :::::'~-, : : : : . . . . . . . . . : :|;;;;;;;;;;;;;;;;;;;;;;;;;;;;¯"~-,

    // ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'\::: : : : : : ::: : :'~-„__ . . . . . . . _„~';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'*~-,

    // ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;'\::::::::: : : : :::: . . .¯¯¯¯¯¯¯¯¯/;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    SharedMemory memory;

    // godgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgodgod!
    [Export] Node players;
    [Export] RigidBody3D ball;

    public void Export(bool done = false, float reward = 0f, bool reset = false) {
        // the scores can be included maybe idk will have to fist myself before i know
        SendToMemory(TeamPositions(1), TeamPositions(2), new Vector2(ball.Position.X, ball.Position.Z), done, reward, reset);
    }

    float[] TeamPositions(int team) {
        float[] teamArray = new float[players.GetChildCount()];

        for (int i = 0; i < teamArray.Length; i += 2) {
            Player player = (Player)(players.GetChild(i)); // yeah bro because casting would totally save us from an error bro
            if (player.team == team) {
                teamArray[i] = player.Position.X;
                teamArray[i + 1] = player.Position.Z;
            }
        }
        
        return teamArray;
    }

    unsafe void SendToMemory(float[] teamAPositions, float[] teamBPositions, Vector2 ballPosition, bool done = false, float reward = 0f, bool reset = false) {
        float[] teamPositions = new float[teamAPositions.Length + teamBPositions.Length];
        Array.Copy(teamAPositions, 0, teamPositions, 0, teamAPositions.Length);
        Array.Copy(teamBPositions, 0, teamPositions, teamAPositions.Length, teamBPositions.Length);

        float[] ballPositionArray = { ballPosition.X, ballPosition.Y };
        float[] positionArray = new float[ballPositionArray.Length + teamPositions.Length];
        Array.Copy(teamPositions, 0, positionArray, 0, teamPositions.Length);
        Array.Copy(ballPositionArray, 0, positionArray, teamPositions.Length, ballPositionArray.Length);
        
        memory.WriteReward(reward);

        if (reset)
        {
            memory.SendResetResults(positionArray);
        }
        else
        {
            memory.SendStepResults(positionArray, reward, done);
        }
    }

    unsafe void Import() {

    }

    public override void _EnterTree() {
        memory = new SharedMemory();
    }

    public float[] ProcessRequests(out bool resetRequested) {
        return memory.ProcessRequests(out resetRequested);
    }

}
