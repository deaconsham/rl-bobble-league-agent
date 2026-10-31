using Godot;
using System;

public partial class Ball : RigidBody3D {
	[Export] float speedThreshold;

	// Highest the ball's centre may go. Arena walls are 1.39 tall (2.7854 * 0.5 scale)
	// and the ball radius is 0.5, so the ball clears a wall only above y = 1.89.
	// 1.2 still allows a small hop but the ball can never get over a wall.
	[Export] float maxHeight = 1.2f;

	public override void _IntegrateForces(PhysicsDirectBodyState3D state) {
		Transform3D t = state.Transform;
		if (t.Origin.Y > maxHeight) {
			t.Origin = new Vector3(t.Origin.X, maxHeight, t.Origin.Z);
			state.Transform = t;
			Vector3 v = state.LinearVelocity;
			if (v.Y > 0) state.LinearVelocity = new Vector3(v.X, 0, v.Z);
		}
	}

	public override void _PhysicsProcess(double delta) {
		if (new Vector2(LinearVelocity.X, LinearVelocity.Z).Length() < speedThreshold) {
			LinearVelocity = new Vector3(0, 0, 0);
			Position = new Vector3(Position.X, 0.5f, Position.Z);
			AngularVelocity = new Vector3(0, 0, 0);
		}
	}
}
