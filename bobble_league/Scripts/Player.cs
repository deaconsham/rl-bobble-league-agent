using Godot;
using System;

public partial class Player : RigidBody3D
{
    [Export] public int team = 1;
    
    [ExportGroup("Physics")] 
    [Export] float launchForce;
    [Export] float kickForceCoefficient; // name implies scaling with speed but not currently
    [Export] float verticallKickForce;
    [Export] float kickSpeedThreshold;
    [Export] float speedThreshold;

    [ExportGroup("Visuals")] 
    [Export] float baseWidth;
    [Export] float baseWidthCoef;
    [Export] float tipSize;


    public Vector3 input;
    // power scale (0.2-1) set by the AI action or the human drag; Launch uses it
    public float launchPower = 1f;

    [Export] Node3D ArrowBase;
    [Export] Node3D ArrowTip;
    bool hidden = true;

    // input is the aim direction; launchPower (0.2-1) sets how hard, so soft and
    // hard shots differ. a caller can override the scale, else the stored one is
    // used (set by the AI action or the human drag). direction only matters for
    // aim; its length is ignored because power is carried separately.
    public void Launch(float powerScale = -1f) {
        float power = powerScale >= 0f ? powerScale : launchPower;
        Vector3 inputNorm = input.Normalized();
        LinearVelocity = new Vector3(inputNorm.X, 0, inputNorm.Z) * launchForce * power;
        
        if (input.Length() > 0) {
            float angle = Mathf.Atan2(input.X, input.Z);
            Rotation = new Vector3(0, angle, 0);
        }
        
        hidden = false;
        ClearArrow();
        
        input = new Vector3(0, 0, 0);
        launchPower = 1f;
    }

    public void DrawArrow() {
        hidden = false;
        ArrowBase.Scale = new Vector3(baseWidth, 1, Mathf.Clamp(input.Length(), 0, 1) * baseWidthCoef);
        ArrowTip.Scale = Vector3.One * tipSize;
        
        float angle = Mathf.Atan2(input.X, input.Z);
        ArrowBase.GlobalRotation = new Vector3(0, angle, 0);
        ArrowTip.GlobalRotation = new Vector3(0, ArrowBase.GlobalRotation.Y + float.DegreesToRadians(180), 0);
        
        ArrowBase.GlobalPosition = Position + (input.Normalized() * Mathf.Clamp(input.Length(), 0, 1)) + input.Normalized() * 0.1f;
        ArrowTip.GlobalPosition = Position + (input.Normalized() * Mathf.Clamp(input.Length(), 0, 1) * 2) + input.Normalized() * 0.6f;
        
        ArrowBase.Visible = true;
        ArrowTip.Visible = true;
    }

    public void ClearArrow() {
        ArrowBase.Visible = false;
        ArrowTip.Visible = false;
        hidden = true;
        
    }

    public override void _Process(double delta) {
        if (hidden == false) {
            DrawArrow();
        }
    }

    void Kick(Node3D node) {
        if (node.IsInGroup("ball")) {
            RigidBody3D ball  = (RigidBody3D)node;
            if ((this.LinearVelocity - ball.LinearVelocity).Length() > kickSpeedThreshold) {
                Vector3 launchVector = LinearVelocity.Normalized() * kickForceCoefficient + new Vector3(0, verticallKickForce, 0);
                ball.ApplyCentralImpulse(launchVector);
            }
        }
    }
    
    public override void _PhysicsProcess(double delta) {
        if (LinearVelocity.Length() < speedThreshold) {
            LinearVelocity = new Vector3(0, 0, 0);
            AngularVelocity = new Vector3(0, 0, 0);
        }
        
    }
}
